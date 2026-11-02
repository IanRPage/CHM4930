"""
Rebuild tracked benchmark CSVs (`results.csv`, `results_runs.csv`, `results_paired.csv`,
`results_by_similarity.csv`) from the promoted runs' saved `predictions/*.csv` files
without modifying `promoted.json`.

From ai/: python drivers/rebuild_results.py
Use this after changing metric or aggregation code when saved predictions are still valid.
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
