import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor

from bioactivity_loader import load_bace1
from evaluation import evaluation_metrics
from featurize.fingerprint import mol_to_ecfp4

N_BITS = 2048
N_ESTIMATORS = 500
RANDOM_STATE = 42
MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.1, 0.3]


def scaffold_split(df):
    """Split molecules 80/10/10 by Bemis-Murcko scaffold."""
    scaffolds = df["smiles"].map(
        lambda smiles: MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    )

    groups = sorted(
        df.groupby(scaffolds).indices.values(),
        key=lambda idx: (-len(idx), idx[0]),
    )

    n = len(df)
    train_idx = []
    val_idx = []
    test_idx = []

    for idx in groups:
        if len(train_idx) + len(idx) <= 0.8 * n:
            train_idx += idx.tolist()
        elif len(val_idx) + len(idx) <= 0.1 * n:
            val_idx += idx.tolist()
        else:
            test_idx += idx.tolist()

    return (
        np.asarray(train_idx),
        np.asarray(val_idx),
        np.asarray(test_idx),
    )


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
    df = load_bace1()

    fingerprints = []

    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(mol_to_ecfp4(mol, N_BITS))

    X = np.stack(fingerprints)
    y = df["pIC50"].to_numpy(dtype=float)
    active = df["active"].to_numpy()

    train_idx, val_idx, test_idx = scaffold_split(df)

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
