"""
Download a published MuFu ensemble from the HuggingFace Hub.

From ai/: python drivers/download_mufu.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mufu.hub import download_version

repo_id = "CHM4930/mufu"
tag = "v1.0.0"
dest = Path.home() / ".local" / "share" / "mufu" / tag


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    download_version(repo_id, tag, dest)
    print(f"MUFU_CHECKPOINT_DIR={dest}")
    print(f"MUFU_MODEL_VERSION={tag}")
