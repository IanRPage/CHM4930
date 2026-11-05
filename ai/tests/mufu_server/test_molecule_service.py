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
    """Check whether a PID is alive, treating reparented Linux zombies as terminated."""
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
    """Await the real child startup marker, surfacing early worker failures instead of hanging."""
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
    """Verify process-boundary validation rejects unsupported chemistry and recovers for carbon-13."""

    async def run():
        """Exercise invalid and valid requests through one service and always close its executor."""
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
    """Require descriptors and coordinates to execute in separate, fully reaped interpreters."""

    async def run():
        """Read recorded worker PIDs after a successful two-phase computation."""
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
    """Keep HTTP health, prediction, and molecule admission responsive during a native GIL stall."""

    class Predictor:
        """Provide a cheap prediction route independent of molecule workers."""

        model_version = "stub"

        def infer(self, smiles):
            """Return observable prediction output while the molecule child is stalled."""
            return {"smiles": smiles}

    async def run():
        """Probe live ASGI routes while a confirmed child remains inside a native call."""
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
    """Require descriptor deadline expiry to terminate its worker before accepting recovery work."""

    async def run():
        """Observe a native descriptor stall, its timeout exception, and reuse of the service slot."""
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
    """Return calculated identifiers and descriptors after terminating an over-budget 3D worker."""

    async def run():
        """Wait for the 3D startup handshake and verify the completed descriptor response survives."""
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
    """Check HTTP timeout mappings and recovery even when the neural model cannot load."""

    def unavailable(_):
        """Represent missing checkpoints without preventing molecule service startup."""
        raise FileNotFoundError("missing checkpoints")

    async def run():
        """Exercise descriptor and 3D deadlines through authenticated ASGI requests."""
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
    """Handle worker crashes, error envelopes, and malformed output without unobserved exceptions."""

    async def run():
        """Collect event-loop errors and verify phase-specific failure handling and process cleanup."""
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
    """Keep draining a native worker despite repeated request cancellation, then release capacity."""

    async def run():
        """Cancel twice while a real worker is active and check cleanup before a recovery request."""
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
    """Terminate active native work on shutdown and reject all subsequent submissions."""

    async def run():
        """Close during a confirmed native stall and observe cancellation only after reaping."""
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
    """Fail closed and reap the subprocess if parent-exit protection cannot be installed."""
    from mufu_server import molecule_service

    processes = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        """Retain the real Popen object so the test can verify its exit status and PID."""
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail(pid):
        """Represent an OS denial while setting up worker lifetime protection."""
        raise OSError("job setup denied")

    monkeypatch.setattr(molecule_service.subprocess, "Popen", record)
    monkeypatch.setattr(molecule_service, "child_job", fail)

    async def run():
        """Observe setup failure through the async API and always drain the executor."""
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
    """Run the review regression molecule that made native InChI hold the server GIL."""
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
        """Keep scheduling loop ticks while the isolated worker processes the circular ladder."""
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


@pytest.mark.parametrize("phase", ["descriptors", "structure"])
def test_worker_diagnostics_reach_stderr_without_exposing_api_keys(
    molecule_worker_command, tmp_path, monkeypatch, capfd, phase
):
    """Preserve worker tracebacks server-side without leaking keys or changing HTTP errors."""

    class Predictor:
        """Supply readiness metadata without loading any neural checkpoints."""

        model_version = "stub"

    key = "worker-diagnostics-test-secret"
    key_file = str(tmp_path / "private-key-file")
    monkeypatch.setenv("MUFU_API_KEY", key)
    monkeypatch.setenv("MUFU_API_KEYS_FILE", key_file)

    async def run():
        """Exercise real worker stderr inheritance and confirm the slot remains reusable."""
        app = create_app(
            Settings(Path("unused"), api_key=key), load=lambda _: Predictor()
        )
        service = app.state.molecules
        command = service._command
        service._command = molecule_worker_command(f"stderr-{phase}")
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            headers = {"X-API-Key": key}
            response = await client.post(
                "/molecule", json={"smiles": "CCO"}, headers=headers
            )
            if phase == "descriptors":
                assert response.status_code == 500
                assert response.json() == {"error": "internal_error"}
            else:
                assert response.status_code == 200
                assert response.json()["formula"] == "C2H6O"
                assert response.json()["molblock"] is None
            pid = json.loads((tmp_path / f"{phase}.json").read_text())["pid"]
            assert not running(pid)
            service._command = command
            recovered = await client.post(
                "/molecule", json={"smiles": "CCO"}, headers=headers
            )
            assert recovered.status_code == 200
            return response.text + recovered.text

    body = asyncio.run(run())
    stdout, stderr = capfd.readouterr()
    assert "controlled molecule worker diagnostic" in stderr
    assert "Traceback" in stderr
    assert "controlled molecule worker diagnostic" not in body + stdout
    assert key not in body + stdout + stderr
    assert key_file not in body + stdout + stderr


def test_abrupt_parent_exit_kills_native_worker(molecule_worker_command, tmp_path):
    """Verify OS parent-exit protection by killing a disposable server during native work."""
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
