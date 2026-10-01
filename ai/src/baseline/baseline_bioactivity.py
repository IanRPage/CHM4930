import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

from evaluation import evaluation_metrics
from featurize.fingerprint import mol_to_ecfp4
from pipeline.combined_loader import load_combined
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
    }


def main():
    df = scaffold_split(load_combined())
    df = df[df["pIC50_BACE1"].notna()].reset_index(drop=True)

    fingerprints = []

    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(mol_to_ecfp4(mol, N_BITS))

    X = np.stack(fingerprints)
    y = df["pIC50_BACE1"].to_numpy(dtype=float)
    active = df["active_BACE1"].to_numpy()

    split = df["split"].to_numpy()
    train_idx, val_idx, test_idx = (
        np.flatnonzero(split == name) for name in ("train", "val", "test")
    )

    best_model = None
    best_max_features = None
    best_val_rmse = float("inf")

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

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_model = model
            best_max_features = max_features

    val_metrics = score_model(
        best_model,
        X,
        y,
        active,
        val_idx,
    )

    test_metrics = score_model(
        best_model,
        X,
        y,
        active,
        test_idx,
    )

    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Best max_features: {best_max_features}")

    print(
        f"Validation: RMSE={val_metrics['rmse']:.4f}, "
        f"R2={val_metrics['r2']:.4f}, "
        f"ROC-AUC={val_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test: RMSE={test_metrics['rmse']:.4f}, "
        f"R2={test_metrics['r2']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
