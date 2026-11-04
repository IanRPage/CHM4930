"""
Download a published MuFu model from the HuggingFace Hub and unpack `ai/runs/`.

From ai/: python drivers/download_mufu.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline.benchmark import PROMOTED_PATH
from mufu.hub import download_version, populate_run_from_download

repo_id = "CHM4930/mufu"
tag = ""  # empty -> promoted["tag"]
populate_run_dir = True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    target_tag = tag or json.loads(PROMOTED_PATH.read_text())["tag"]
    dest = Path.home() / ".local" / "share" / "mufu" / target_tag
    download_version(repo_id, target_tag, dest)
    print(f"MUFU_CHECKPOINT_DIR={dest}")
    print(f"MUFU_MODEL_VERSION={target_tag}")
    if populate_run_dir:
        run_dir = populate_run_from_download(dest, target_tag)
        print(f"Populated run folder: {run_dir}")
        if not list((run_dir / "predictions").glob("val_seed_*.csv")):
            print(
                "Note: this tag did not bundle predictions/*.csv; run "
                "`python drivers/rescore_mufu.py` to populate local predictions."
            )
