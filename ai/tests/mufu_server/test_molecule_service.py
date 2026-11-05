import asyncio
import ctypes
import gc
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from rdkit import Chem

from mufu_server.app import create_app
from mufu_server.config import Settings
from mufu_server.molecule import InvalidMoleculeSmiles
from mufu_server.molecule_service import MoleculeBusy, MoleculeService, WorkerFailure


def running(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # After abrupt parent death, init/subreaper owns reaping. Zombies are dead.
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().split(") ", 1)[1][0] != "Z"


async def marker(path, task):
    async with asyncio.timeout(10):
        while not path.exists():
            if task.done():
                task.result()
                pytest.fail("worker exited without the startup handshake")
            await asyncio.sleep(0.01)
    data = json.loads(path.read_text())
    assert data["pid"] != os.getpid()
    return data["pid"]


@pytest.mark.parametrize("smiles", ["*", "*C", "[999C]"])
def test_real_worker_rejects_unsupported_chemistry(smiles):
    async def run():
        service = MoleculeService()
        try:
            with pytest.raises(InvalidMoleculeSmiles):
                await service.compute(smiles, 10)
            assert (await service.compute("[13CH4]", 10)).exact_mass == pytest.approx(
                17.034654968
            )
        finally:
            await service.close()

    asyncio.run(run())


def test_phases_use_distinct_reaped_processes(molecule_worker_command, tmp_path):
    async def run():
        service = MoleculeService(command=molecule_worker_command("trace"))
        try:
            result = await service.compute("CCO", 10)
            pids = [
                json.loads((tmp_path / f"{phase}.json").read_text())["pid"]
                for phase in ("descriptors", "structure")
            ]
            assert len(set(pids + [os.getpid()])) == 3
            assert all(not running(pid) for pid in pids)
            assert result.molblock is not None
        finally:
            await service.close()

    asyncio.run(run())


def test_native_stall_does_not_block_predict_health_or_rate_gate(
    molecule_worker_command, tmp_path
):
    class Predictor:
        model_version = "stub"

        def infer(self, smiles):
            return {"smiles": smiles}

    async def run():
        app = create_app(
            Settings(Path("unused"), api_key="test", request_timeout_s=30),
            load=lambda _: Predictor(),
        )
        service = app.state.molecules
        service._command = molecule_worker_command("hang-descriptors")
        async with app.router.lifespan_context(app):
            while not app.state.loaded.is_set():
                await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                headers = {"X-API-Key": "test"}
                task = asyncio.create_task(
                    client.post("/molecule", json={"smiles": "CCO"}, headers=headers)
                )
                try:
                    pid = await marker(tmp_path / "descriptors.json", task)
                    assert running(pid)
                    # Child stays inside a 60-second GIL-holding native call.
                    # These responses must complete while that child is alive.
                    async with asyncio.timeout(5):
                        health = await client.get("/health")
                        predict = await client.post(
                            "/predict", json={"smiles": "CCO"}, headers=headers
                        )
                        busy = await client.post(
                            "/molecule", json={"smiles": "CCO"}, headers=headers
                        )
                    assert health.status_code == predict.status_code == 200
                    assert busy.status_code == 503 and busy.json() == {"error": "busy"}
                    assert running(pid) and not task.done()
                finally:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert not running(pid)

    asyncio.run(run())


def test_descriptor_deadline_kills_reaps_and_releases_capacity(
    molecule_worker_command, tmp_path
):
    async def run():
        service = MoleculeService(command=molecule_worker_command("hang-descriptors"))
        try:
            task = asyncio.create_task(service.compute("CCO", 2))
            pid = await marker(tmp_path / "descriptors.json", task)
            assert running(pid)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 10)
            assert not running(pid)
            service._command = molecule_worker_command("trace")
            assert (await service.compute("CCO", 10)).formula == "C2H6O"
        finally:
            await service.close()

    asyncio.run(run())


def test_structure_deadline_preserves_descriptors_and_reaps(
    molecule_worker_command, tmp_path
):
    async def run():
        service = MoleculeService(command=molecule_worker_command("hang-structure"))
        try:
            task = asyncio.create_task(service.compute("CCO", 10))
            pid = await marker(tmp_path / "structure.json", task)
            assert running(pid)
            result = await asyncio.wait_for(task, 10)
            assert result.molecular_weight == pytest.approx(46.069)
            assert result.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
            assert result.molblock is None
            assert not running(pid)
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["descriptors", "structure"])
def test_http_deadlines_reap_workers_and_allow_recovery(
    molecule_worker_command, tmp_path, phase
):
    def unavailable(_):
        raise FileNotFoundError("missing checkpoints")

    async def run():
        app = create_app(
            Settings(
                Path("unused"),
                api_key="test",
                request_timeout_s=3 if phase == "descriptors" else 10,
            ),
            load=unavailable,
        )
        service = app.state.molecules
        command = service._command
        service._command = molecule_worker_command(f"hang-{phase}")
        async with app.router.lifespan_context(app):
            while not app.state.loaded.is_set():
                await asyncio.sleep(0.01)
            assert isinstance(app.state.load_error, FileNotFoundError)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                headers = {"X-API-Key": "test"}
                task = asyncio.create_task(
                    client.post("/molecule", json={"smiles": "CCO"}, headers=headers)
                )
                try:
                    pid = await marker(tmp_path / f"{phase}.json", task)
                    assert running(pid)
                    response = await asyncio.wait_for(asyncio.shield(task), 15)
                finally:
                    if not task.done():
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                assert not running(pid)
                if phase == "descriptors":
                    assert response.status_code == 503
                    assert response.json() == {"error": "timeout"}
                else:
                    assert response.status_code == 200
                    assert response.json()["formula"] == "C2H6O"
                    assert response.json()["molblock"] is None
                service._command = command
                recovered = await client.post(
                    "/molecule", json={"smiles": "CCO"}, headers=headers
                )
                assert recovered.status_code == 200
                assert recovered.json()["formula"] == "C2H6O"
                assert recovered.json()["molblock"] is not None

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["descriptors", "structure"])
@pytest.mark.parametrize("failure", ["crash", "error", "malformed"])
def test_worker_failures_are_observed_and_reaped(
    molecule_worker_command, tmp_path, phase, failure
):
    async def run():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: unhandled.append(context)
        )
        service = MoleculeService(command=molecule_worker_command(f"{failure}-{phase}"))
        try:
            if phase == "descriptors":
                with pytest.raises(WorkerFailure):
                    await service.compute("CCO", 10)
            else:
                result = await service.compute("CCO", 10)
                assert result.formula == "C2H6O" and result.molblock is None
            pid = json.loads((tmp_path / f"{phase}.json").read_text())["pid"]
            assert not running(pid)
        finally:
            await service.close()
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["descriptors", "structure"])
def test_repeated_cancellation_drains_worker(molecule_worker_command, tmp_path, phase):
    async def run():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: unhandled.append(context)
        )
        service = MoleculeService(command=molecule_worker_command(f"hang-{phase}"))
        try:
            task = asyncio.create_task(service.compute("CCO", 30))
            pid = await marker(tmp_path / f"{phase}.json", task)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not running(pid)
            service._command = molecule_worker_command("trace")
            assert (await service.compute("CCO", 10)).formula == "C2H6O"
        finally:
            await service.close()
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled

    asyncio.run(run())


def test_service_shutdown_kills_active_worker(molecule_worker_command, tmp_path):
    async def run():
        service = MoleculeService(command=molecule_worker_command("hang-descriptors"))
        task = asyncio.create_task(service.compute("CCO", 30))
        try:
            pid = await marker(tmp_path / "descriptors.json", task)
        finally:
            await service.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not running(pid)
        with pytest.raises(MoleculeBusy):
            await service.compute("CCO", 10)

    asyncio.run(run())


def test_failed_job_setup_kills_child(monkeypatch):
    from mufu_server import molecule_service

    processes = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail(pid):
        raise OSError("job setup denied")

    monkeypatch.setattr(molecule_service.subprocess, "Popen", record)
    monkeypatch.setattr(molecule_service, "child_job", fail)

    async def run():
        service = MoleculeService()
        try:
            with pytest.raises(WorkerFailure):
                await service.compute("CCO", 10)
        finally:
            await service.close()

    asyncio.run(run())
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert not running(processes[0].pid)


def test_real_large_inchi_input_is_isolated():
    # The review's 934-character circular ladder: InChI holds the GIL in-process.
    mol = Chem.RWMol()
    for _ in range(320):
        mol.AddAtom(Chem.Atom("C"))
    for i in range(160):
        for a, b in ((i, (i + 1) % 160), (i + 160, (i + 1) % 160 + 160), (i, i + 160)):
            mol.AddBond(a, b, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol)
    assert len(smiles) <= 1000

    async def run():
        service = MoleculeService()
        ticks = 0
        task = asyncio.create_task(service.compute(smiles, 10))
        try:
            while not task.done():
                ticks += 1
                await asyncio.sleep(0.01)
            result = await task
            assert result.formula == "C320H320"
            assert result.inchi_key is not None and result.molblock is None
            assert ticks > 1
        finally:
            await service.close()

    asyncio.run(run())


def test_abrupt_parent_exit_kills_native_worker(molecule_worker_command, tmp_path):
    command = molecule_worker_command("hang-descriptors")
    code = f"""import asyncio
from mufu_server.molecule_service import MoleculeService
async def main():
    service=MoleculeService(command={command!r})
    await service.compute('CCO', 30)
asyncio.run(main())
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    pid = None
    with subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as parent:
        try:
            end = time.monotonic() + 10
            marker_path = tmp_path / "descriptors.json"
            while not marker_path.exists() and time.monotonic() < end:
                assert parent.poll() is None
                time.sleep(0.01)
            pid = json.loads(marker_path.read_text())["pid"]
            assert running(pid)
            parent.kill()
            parent.wait(5)
            end = time.monotonic() + 5
            while running(pid) and time.monotonic() < end:
                time.sleep(0.01)
            assert not running(pid)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(5)
            if pid is not None and running(pid):
                os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
