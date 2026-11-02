"""
Promote a complete MuFu run in `ai/runs/<run_id>/` to be the tracked default run.

From ai/: python drivers/promote_mufu.py
Scores the held-out `test` predictions (`test_metrics.csv`) and rebuilds
`ai/data/baseline/promoted.json`, `results.csv`, `results_runs.csv`,
`results_paired.csv`, and `results_by_similarity.csv` without retraining.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from baseline.benchmark import PROMOTED_PATH, RUNS_DIR, promote_run
from baseline.compare import PAIRED_PATH
from baseline.results import summarize_seeds

# set `run_id` to winning run in `ai/runs/<run_id>/` before running
run_id = ""
tag: str | None = None  # None -> keep promoted["tag"]; set e.g. "v1.1.0" to bump


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")
    if not run_id:
        raise SystemExit(
            "Set `run_id` at the top of drivers/promote_mufu.py to the run you want to promote "
            "(or run `python drivers/rebuild_results.py` to rebuild tracked CSVs without changing promoted.json)."
        )

    run_dir = RUNS_DIR / run_id
    all_rows = promote_run(run_dir, tag=tag)
    resolved_tag = json.loads(PROMOTED_PATH.read_text()).get("tag", tag)
    run_rows = all_rows[all_rows["run_id"] == run_id]
    summary = summarize_seeds(run_rows[run_rows["split"] == "test"])
    cols = [
        "dataset",
        "endpoint",
        "metric",
        "value",
        "std",
        "ci_low",
        "ci_high",
        "n_seeds",
    ]
    print(f"Promoted {run_id} (tag={resolved_tag}).\n\nSingle-model test summary:")
    print(summary[cols].round(4).to_string(index=False))

    ens_cov_path = run_dir / "ensemble_coverage.csv"
    if ens_cov_path.exists():
        print("\nEnsemble test interval coverage:")
        print(pd.read_csv(ens_cov_path).round(4).to_string(index=False))

    if PAIRED_PATH.exists():
        paired = pd.read_csv(PAIRED_PATH)
        print("\nTest paired bootstrap (mufu - rf):")
        print(paired[paired["split"] == "test"].round(4).to_string(index=False))
