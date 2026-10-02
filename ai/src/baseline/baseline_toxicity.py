"""Random forest toxicity baseline on ECFP4 fingerprints.

Run from ``ai/`` with ``PYTHONPATH=src``:

    python -m baseline.baseline_toxicity
"""

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier

from evaluation import evaluation_metrics
from featurize.fingerprint import mol_to_ecfp4
from pipeline.combined_loader import TOX_TASKS, load_combined
from pipeline.preprocess import summarize_labels
from pipeline.split import scaffold_split
from pipeline.toxicity_loader import DATASETS

N_BITS = 2048
N_ESTIMATORS = 500
RANDOM_STATE = 42
MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.1, 0.3]


def positive_probabilities(model, X):
    if len(X) == 0:
        return np.empty(0)
    positive_column = np.flatnonzero(model.classes_ == 1)[0]
    return model.predict_proba(X)[:, positive_column]


def fit_endpoint(X, y, train_idx, val_idx):
    """Fit using observed training/validation indices; never inspect test labels."""
    if len(train_idx) == 0:
        return None, "skipped: no observed training labels"
    if np.unique(y[train_idx]).size < 2:
        return None, "skipped: training labels contain only one class"

    selectable = np.unique(y[val_idx]).size >= 2
    options = MAX_FEATURES_OPTIONS if selectable else ["sqrt"]
    best_model, best_auc = None, -np.inf
    for max_features in options:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X[train_idx], y[train_idx])
        val_auc = evaluation_metrics(
            y[val_idx],
            positive_probabilities(model, X[val_idx]),
            task_type="classification",
        )["roc_auc"]
        if best_model is None or val_auc > best_auc:
            best_model, best_auc = model, val_auc

    status = (
        "selected by validation ROC-AUC"
        if selectable
        else "validation selection unavailable: no labels or one class; using sqrt"
    )
    return best_model, status


def main():
    df = scaffold_split(load_combined())
    df = df.loc[df[TOX_TASKS].notna().any(axis=1)].reset_index(drop=True)
    X = (
        np.stack([mol_to_ecfp4(Chem.MolFromSmiles(s), N_BITS) for s in df["smiles"]])
        if len(df)
        else np.empty((0, N_BITS), dtype=np.uint8)
    )
    y = df[TOX_TASKS].to_numpy(dtype=float)
    observed = df[TOX_TASKS].notna().to_numpy()
    split = df["split"].to_numpy()
    indices = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}
    predictions = {
        name: np.full((len(indices[name]), len(TOX_TASKS)), np.nan)
        for name in ("val", "test")
    }
    trained = np.zeros(len(TOX_TASKS), dtype=bool)
    selections = []

    for name, idx in indices.items():
        print(f"\n{name}: {len(idx)} molecules with at least one toxicity label")
        print(summarize_labels(df.iloc[idx], TOX_TASKS).round(4).to_string())

    for j, task in enumerate(TOX_TASKS):
        train_idx, val_idx = (
            indices[name][observed[indices[name], j]] for name in ("train", "val")
        )
        model, status = fit_endpoint(X, y[:, j], train_idx, val_idx)
        selections.append(
            (task, model.max_features if model is not None else None, status)
        )
        if model is None:
            continue
        trained[j] = True
        for name, scores in predictions.items():
            scores[:, j] = positive_probabilities(model, X[indices[name]])

    metrics = {}
    for name, scores in predictions.items():
        idx = indices[name]
        # Untrained endpoints have no predictions and must not enter macro means.
        mask = observed[idx] & trained
        metrics[name] = evaluation_metrics(y[idx], scores, "multitask", mask=mask)
        groups = {"overall": list(range(len(TOX_TASKS)))}
        groups.update(
            {
                dataset: [TOX_TASKS.index(task) for task in config["tasks"]]
                for dataset, config in DATASETS.items()
            }
        )
        for group, columns in groups.items():
            result = (
                metrics[name]
                if group == "overall"
                else evaluation_metrics(
                    y[idx][:, columns],
                    scores[:, columns],
                    "multitask",
                    mask=mask[:, columns],
                )
            )
            count = np.isfinite(result["per_task_roc_auc"]).sum()
            print(
                f"{name} {group}: macro ROC-AUC={result['mean_roc_auc']:.4f} "
                f"({count}/{len(columns)} evaluable endpoints)"
            )

    print("\nEndpoint results (FDA_APPROVED predicts approval status):")
    for j, (task, max_features, status) in enumerate(selections):
        print(
            f"{task}: max_features={max_features}, "
            f"val ROC-AUC={metrics['val']['per_task_roc_auc'][j]:.4f}, "
            f"test ROC-AUC={metrics['test']['per_task_roc_auc'][j]:.4f}; {status}"
        )


if __name__ == "__main__":
    main()
