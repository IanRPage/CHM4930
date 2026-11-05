"""Controlled native stalls/crashes for process-supervision integration tests."""

import ctypes
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from mufu_server.process_lifetime import arm_parent_exit


def main():
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
