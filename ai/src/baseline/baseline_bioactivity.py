"""
Random forest pIC50 baselines for BACE-1 and EGFR on ECFP4 fingerprints, one model per target.

How to run (from `ai/`, with `PYTHONPATH=src`):

    python -m baseline.baseline_bioactivity
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from baseline.benchmark import SEEDS, load_benchmark_split, run_benchmark
from evaluation import evaluation_metrics
from featurize.fingerprint import FINGERPRINT_BITS, mol_to_ecfp4
from pipeline.combined_loader import TARGETS
from pipeline.preprocess import to_mol

N_ESTIMATORS = 500
MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.1, 0.3]


def prepare_target(df, target):
    col = f"pIC50_{target}"
    labeled = df[col].notna().to_numpy()
    sub = df.loc[labeled].reset_index(drop=True)

    fingerprints = [mol_to_ecfp4(to_mol(s), FINGERPRINT_BITS) for s in sub["smiles"]]
    X = np.stack(fingerprints)
    y = sub[col].to_numpy(dtype=float)

    split = sub["split"].to_numpy()
    train_idx, val_idx, test_idx = (
        np.flatnonzero(split == name) for name in ("train", "val", "test")
    )
    split_masks = {
        name: df.loc[df["split"] == name, col].notna().to_numpy()
        for name in ("val", "test")
    }
    return {
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_masks": split_masks,
    }


def select_max_features(data, seed=None):
    if seed is None:
        seed = SEEDS[0]
    X, y = data["X"], data["y"]
    train_idx, val_idx = data["train_idx"], data["val_idx"]

    best_max_features = None
    best_rmse = float("inf")

    for max_features in MAX_FEATURES_OPTIONS:
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=max_features,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        val_rmse = evaluation_metrics(
            y[val_idx],
            model.predict(X[val_idx]),
            task_type="regression",
        )["rmse"]
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_max_features = max_features

    return best_max_features


def main():
    df = load_benchmark_split()
    prepared = {target: prepare_target(df, target) for target in TARGETS}
    selected = {
        target: select_max_features(prepared[target], seed=SEEDS[0])
        for target in TARGETS
    }

    def fit_predict(split_df, seed):
        predictions = {"val": {}, "test": {}}
        for target in TARGETS:
            data = prepared[target]
            X, y = data["X"], data["y"]
            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_features=selected[target],
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(X[data["train_idx"]], y[data["train_idx"]])
            col = f"pIC50_{target}"
            for split, idx_key in (("val", "val_idx"), ("test", "test_idx")):
                mask = data["split_masks"][split]
                scores = np.full(len(mask), np.nan, dtype=float)
                scores[mask] = model.predict(X[data[idx_key]])
                predictions[split][col] = scores
        return predictions

    params = {"max_features": selected, "n_estimators": N_ESTIMATORS}
    summary = run_benchmark(
        fit_predict,
        model="rf_bioactivity",
        encoder=f"ecfp4-{FINGERPRINT_BITS}",
        params=params,
        seeds=SEEDS,
        df=df,
    )

    for target in TARGETS:
        data = prepared[target]
        endpoint = f"pIC50_{target}"
        sub = summary[summary["endpoint"] == endpoint].set_index(["split", "metric"])
        print(f"\n{target}")
        print(f"Training samples: {len(data['train_idx'])}")
        print(f"Validation samples: {len(data['val_idx'])}")
        print(f"Test samples: {len(data['test_idx'])}")
        print(f"Best max_features: {selected[target]}")
        for label, split in (("Validation", "val"), ("Test", "test")):
            print(
                f"{label}: "
                f"RMSE={sub.loc[(split, 'rmse'), 'value']:.4f}, "
                f"R2={sub.loc[(split, 'r2'), 'value']:.4f}, "
                f"ROC-AUC={sub.loc[(split, 'roc_auc'), 'value']:.4f}"
            )


if __name__ == "__main__":
    main()
