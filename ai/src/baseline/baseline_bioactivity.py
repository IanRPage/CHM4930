"""
Random forest pIC50 baselines for BACE-1 and EGFR on ECFP4 fingerprints, one model per target.

How to run (from `ai/`, with `PYTHONPATH=src`):

    python -m baseline.baseline_bioactivity
"""

import json

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

from baseline.results import record_results
from evaluation import evaluation_metrics
from featurize.fingerprint import mol_to_ecfp4
from pipeline.combined_loader import TARGETS, load_combined
from pipeline.split import scaffold_split

N_BITS = 2048
N_ESTIMATORS = 500
RANDOM_STATE = 42
MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.1, 0.3]


def score_model(model, X, y, active, idx):
    predictions = model.predict(X[idx])

    regression = evaluation_metrics(
        y[idx],
        predictions,
        task_type="regression",
    )

    classification = evaluation_metrics(
        active[idx],
        predictions,
        task_type="classification",
    )

    return {
        "rmse": regression["rmse"],
        "r2": regression["r2"],
        "roc_auc": classification["roc_auc"],
        "n_labeled": len(idx),
        "n_positive": int(np.sum(active[idx] == 1)),
    }


def result_rows(target, split_metrics, max_features):
    params = json.dumps({"max_features": max_features, "n_estimators": N_ESTIMATORS})
    return [
        {
            "encoder": f"ecfp4-{N_BITS}",
            "dataset": target.lower(),
            "endpoint": f"pIC50_{target}",
            "split": split,
            "metric": metric,
            "value": metrics[metric],
            "n_labeled": metrics["n_labeled"],
            "n_positive": metrics["n_positive"],
            "params": params,
        }
        for split, metrics in split_metrics.items()
        for metric in ("rmse", "r2", "roc_auc")
    ]


def fit_target(df, target):
    df = df[df[f"pIC50_{target}"].notna()].reset_index(drop=True)

    fingerprints = []

    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(mol_to_ecfp4(mol, N_BITS))

    X = np.stack(fingerprints)
    y = df[f"pIC50_{target}"].to_numpy(dtype=float)
    active = df[f"active_{target}"].to_numpy()

    split = df["split"].to_numpy()
    train_idx, val_idx, test_idx = (
        np.flatnonzero(split == name) for name in ("train", "val", "test")
    )

    best_model = None
    best_max_features = None
    best_val_metrics = {"rmse": float("inf")}

    for max_features in MAX_FEATURES_OPTIONS:
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        model.fit(X[train_idx], y[train_idx])

        val_metrics = score_model(
            model,
            X,
            y,
            active,
            val_idx,
        )

        if val_metrics["rmse"] < best_val_metrics["rmse"]:
            best_val_metrics = val_metrics
            best_model = model
            best_max_features = max_features

    test_metrics = score_model(
        best_model,
        X,
        y,
        active,
        test_idx,
    )

    print(f"\n{target}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Best max_features: {best_max_features}")

    print(
        f"Validation: RMSE={best_val_metrics['rmse']:.4f}, "
        f"R2={best_val_metrics['r2']:.4f}, "
        f"ROC-AUC={best_val_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test: RMSE={test_metrics['rmse']:.4f}, "
        f"R2={test_metrics['r2']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}"
    )

    return result_rows(
        target, {"val": best_val_metrics, "test": test_metrics}, best_max_features
    )


def main():
    df = scaffold_split(load_combined())
    rows = [row for target in TARGETS for row in fit_target(df, target)]
    record_results(rows, model="rf_bioactivity")


if __name__ == "__main__":
    main()
