"""Controlled native stalls/crashes for process-supervision integration tests."""

import ctypes
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from mufu_server.process_lifetime import arm_parent_exit


def main():
    """Run a real worker phase with an optional native stall or protocol fault.

    Write the child PID only after the input handshake, allowing tests to observe
    an active isolated worker before asserting timeout or cancellation cleanup.
    """
    mode, directory, phase, parent_pid = sys.argv[1:]
    arm_parent_exit(int(parent_pid))
    line = sys.stdin.buffer.readline()
    if not line:
        return
    smiles = json.loads(line)["smiles"]
    path = Path(directory) / f"{phase}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "phase": phase}))
    temporary.replace(path)

    if mode == f"stderr-{phase}":
        assert "MUFU_API_KEY" not in os.environ
        assert "MUFU_API_KEYS_FILE" not in os.environ
        try:
            raise RuntimeError("controlled molecule worker diagnostic")
        except RuntimeError:
            logging.getLogger(__name__).exception("molecule worker failed")
        print(json.dumps({"error": "internal_error"}))
        return

    if mode == f"hang-{phase}":
        # PyDLL deliberately holds this interpreter's GIL during a native call.
        # A thread-only supervisor would freeze its own event loop here.
        if os.name == "nt":
            ctypes.PyDLL("kernel32").Sleep(60_000)
        else:
            ctypes.PyDLL(None).sleep(60)
    if mode == f"crash-{phase}":
        os._exit(7)
    if mode == f"malformed-{phase}":
        print("not-json")
        return
    if mode == f"error-{phase}":
        print(json.dumps({"error": "internal_error"}))
        return

    from mufu_server.molecule import molecule_info, molecule_molblock

    result = (
        asdict(molecule_info(smiles, include_3d=False))
        if phase == "descriptors"
        else molecule_molblock(smiles)
    )
    print(json.dumps({"result": result}))
    assert "torch" not in sys.modules


if __name__ == "__main__":
    main()
