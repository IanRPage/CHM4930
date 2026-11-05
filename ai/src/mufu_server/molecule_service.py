"""Bounded process supervision. No RDKit operation runs on a server thread."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from mufu_server.molecule import InvalidMoleculeSmiles, MoleculeInfo
from mufu_server.process_lifetime import child_job

log = logging.getLogger(__name__)
STRUCTURE_TIMEOUT_S = 2.0  # includes fresh interpreter startup and RDKit imports
POLL_S = 0.05
WORKER_COMMAND = (sys.executable, "-u", "-m", "mufu_server.molecule_worker")


class MoleculeBusy(Exception):
    """The service is closed or its single request slot is already occupied."""


class WorkerFailure(Exception):
    """A worker failed to start, exited abnormally, or returned invalid output."""


class WorkerCancelled(Exception):
    """The supervisor observed cancellation and finished subprocess cleanup."""


async def _drain(job) -> None:
    """Wait for cleanup and retrieve the job's exception despite cancellation.

    Shielding keeps repeated request or shutdown cancellations from abandoning
    the supervisor before it has terminated and reaped its worker.
    """
    while not job.done():
        try:
            await asyncio.shield(job)
        except asyncio.CancelledError:
            continue
        except Exception:
            log.debug("molecule worker failed during cleanup", exc_info=True)
            break
    if not job.cancelled():
        job.exception()


class MoleculeService:
    """Admit one molecule request and supervise its sequential native workers.

    Only pipe I/O and process management run in the single executor thread.
    RDKit calculations execute in fresh child interpreters, independently of
    the prediction executor and neural-model state.
    """

    def __init__(self, *, command=WORKER_COMMAND):
        """Create an idle supervisor with no subprocesses or queued requests.

        ``command`` selects the worker bootstrap; tests substitute real child
        programs that reproduce stalls and failures.
        """
        self._command = tuple(command)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="molecule-supervisor"
        )
        self._busy = threading.Lock()
        self._closed = False
        self._job = None
        self._cancel = None

    async def compute(self, smiles: str, timeout: float) -> MoleculeInfo:
        """Compute within an admission-time budget without blocking the loop.

        Raise MoleculeBusy immediately if occupied or closed. Descriptor timeout
        raises TimeoutError; optional 3D failure preserves completed descriptors.
        Cancellation drains subprocess cleanup before releasing the request slot.
        """
        if self._closed or not self._busy.acquire(blocking=False):
            raise MoleculeBusy
        cancel = threading.Event()
        self._cancel = cancel
        try:
            job = asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._compute,
                smiles,
                time.monotonic() + timeout,
                cancel,
            )
            self._job = job
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                cancel.set()
                await _drain(job)
                raise
            except WorkerCancelled:
                raise asyncio.CancelledError from None
        finally:
            self._job = self._cancel = None
            self._busy.release()

    async def close(self) -> None:
        """Reject new requests, cancel active work, and join the drained supervisor.

        Completion is observed before shutting down the executor so shutdown
        cannot leave native chemistry running in an abandoned child.
        """
        self._closed = True
        if self._job is not None:
            self._cancel.set()
            await _drain(self._job)
        # No work remains: joining the single supervisor cannot wait on RDKit.
        self._executor.shutdown(wait=True)

    def _compute(self, smiles, deadline, cancel):
        """Run descriptors first, then optional 3D within the remaining budget.

        ``deadline`` is an absolute monotonic timestamp shared by both phases.
        Structure work also has its own shorter limit; its timeout or failure
        cannot discard descriptors already returned by the first worker.
        """
        data = self._phase("descriptors", smiles, deadline, cancel)
        if not isinstance(data, dict):
            raise WorkerFailure("invalid descriptor response")
        info = MoleculeInfo(**data)
        if cancel.is_set():
            raise WorkerCancelled
        if time.monotonic() >= deadline:
            return info
        try:
            block = self._phase(
                "structure",
                info.smiles,
                min(deadline, time.monotonic() + STRUCTURE_TIMEOUT_S),
                cancel,
            )
            if block is not None and not isinstance(block, str):
                raise WorkerFailure("invalid structure response")
        except (WorkerFailure, TimeoutError, InvalidMoleculeSmiles):
            log.warning("molecule structure unavailable; returning descriptors")
            block = None
        return replace(info, molblock=block)

    def _phase(self, phase, smiles, deadline, cancel):
        """Exchange one JSON result with a deadline-controlled child interpreter.

        Install parent-exit protection before sending SMILES, then poll for
        completion or cancellation. All exit paths terminate and reap the child
        before returning. Worker stderr inherits the server's sink; API keys
        are removed from the child environment and diagnostics stay off stdout.
        """
        if cancel.is_set():
            raise WorkerCancelled
        if time.monotonic() >= deadline:
            raise TimeoutError
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        # Chemistry workers need neither the service key nor the key-file path.
        env.pop("MUFU_API_KEY", None)
        env.pop("MUFU_API_KEYS_FILE", None)
        payload = (json.dumps({"smiles": smiles}) + "\n").encode()
        try:
            with subprocess.Popen(
                [*self._command, phase, str(os.getpid())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ) as process:
                try:
                    with child_job(process.pid):
                        while True:
                            if cancel.is_set():
                                raise WorkerCancelled
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError
                            try:
                                output, _ = process.communicate(
                                    payload, timeout=min(POLL_S, remaining)
                                )
                                break
                            except subprocess.TimeoutExpired:
                                payload = (
                                    None  # communicate may be resumed, not re-sent
                                )
                        if cancel.is_set():
                            raise WorkerCancelled
                        if time.monotonic() > deadline:
                            raise TimeoutError
                        if process.returncode:
                            raise WorkerFailure(
                                f"{phase} worker exited {process.returncode}"
                            )
                finally:
                    # Cancellation of a Future does NOT terminate native work.
                    # Kill and reap before releasing capacity, including failed
                    # job setup, timeout, client cancellation, and shutdown.
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
        except TimeoutError:
            # TimeoutError is an OSError subclass, but denotes the request
            # deadline rather than a failure to start or supervise the worker.
            raise
        except OSError as exc:
            raise WorkerFailure(f"cannot run {phase} worker") from exc
        try:
            message = json.loads(output)
            error = message.get("error")
            result = message.get("result")
            if error is None and "result" not in message:
                raise ValueError("missing result")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise WorkerFailure(f"invalid {phase} worker response") from exc
        if error == "invalid_smiles":
            raise InvalidMoleculeSmiles
        if error is not None:
            raise WorkerFailure(f"{phase} worker failed")
        return result
