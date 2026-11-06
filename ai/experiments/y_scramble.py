"""
Y-scrambling control for the RF pIC50 baseline: refit with train labels shuffled and
check test performance collapses to chance. Default split, seed 0, the `max_features`
`results.csv` records for each target, `N_PERMUTATIONS` independent shuffles. Nothing is
written to `results.csv`.

    python experiments/y_scramble.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from baseline.baseline_bioactivity import N_ESTIMATORS, prepare_target
from baseline.benchmark import load_benchmark_split
from evaluation import evaluation_metrics
from pipeline.combined_loader import TARGETS

SEED = 0
N_PERMUTATIONS = 5
MAX_FEATURES = {"BACE1": 0.1, "EGFR": "log2"}


def score_test_split(data, y_train) -> dict:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES[data["target"]],
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(data["X"][data["train_idx"]], y_train)
    y_test = data["y"][data["test_idx"]]
    return evaluation_metrics(
        y_test, model.predict(data["X"][data["test_idx"]]), task_type="regression"
    )


if __name__ == "__main__":
    df = load_benchmark_split()
    rng = np.random.default_rng(SEED)
    rows = []
    for target in TARGETS:
        data = {**prepare_target(df, target), "target": target}
        y_train = data["y"][data["train_idx"]]
        runs = [("real", y_train)] + [
            (f"scrambled-{k}", rng.permutation(y_train)) for k in range(N_PERMUTATIONS)
        ]
        for labels, y in runs:
            rows.append(
                {"target": target, "labels": labels, **score_test_split(data, y)}
            )
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    scrambled = out[out["labels"] != "real"]
    print("\nscrambled mean +/- std")
    print(
        scrambled.groupby("target")[["rmse", "r2", "spearman"]]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )
