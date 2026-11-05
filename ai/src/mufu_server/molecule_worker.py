"""One request, one phase, one fresh process. Never import the model server here."""

import json
import logging
import sys
from dataclasses import asdict

from mufu_server.process_lifetime import arm_parent_exit

log = logging.getLogger(__name__)


def main() -> None:
    phase, parent_pid = sys.argv[1:]
    arm_parent_exit(int(parent_pid))
    # The parent installs Windows job protection before sending this line.
    # EOF before that handshake exits without importing or running chemistry.
    line = sys.stdin.buffer.readline()
    if not line:
        return
    smiles = json.loads(line)["smiles"]

    from mufu_server.molecule import (
        InvalidMoleculeSmiles,
        molecule_info,
        molecule_molblock,
    )

    try:
        if phase == "descriptors":
            result = asdict(molecule_info(smiles, include_3d=False))
        elif phase == "structure":
            result = molecule_molblock(smiles)
        else:
            raise ValueError("unknown molecule phase")
        response = {"result": result}
    except InvalidMoleculeSmiles:
        response = {"error": "invalid_smiles"}
    except Exception:
        log.exception("molecule worker failed")
        response = {"error": "internal_error"}
    json.dump(response, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
