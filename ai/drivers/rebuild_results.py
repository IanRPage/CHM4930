"""
Rebuild `ai/data/baseline/*.csv` from the promoted runs' saved predictions.

From ai/: python drivers/rebuild_results.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline.benchmark import PROMOTED_PATH, RESULTS_PATH, rebuild_tracked_results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("This script takes no arguments.")

    all_rows = rebuild_tracked_results()
    print(
        f"Rebuilt {len(all_rows)} rows in {RESULTS_PATH} "
        f"from {PROMOTED_PATH} (val n_boot=0, test n_boot=2000)."
    )
