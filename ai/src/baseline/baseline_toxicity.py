"""
Random forest toxicity baseline on ECFP4 fingerprints.

Run from `ai/` with `PYTHONPATH=src`:

    python -m baseline.baseline_toxicity

Runs on every scaffold split in `SPLIT_IDS`, tuning `max_features` on each split's own
validation set.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from baseline import benchmark
from baseline.benchmark import (
    SEEDS,
    SPLIT_IDS,
    load_benchmark_split,
    missing_promoted_runs,
    promote_run,
    rebuild_tracked_results,
    run_benchmark,
    score_run_predictions,
)
from baseline.results import summarize_seeds
from evaluation import evaluation_metrics
from featurize.fingerprint import FINGERPRINT_BITS, mol_to_ecfp4
from pipeline.combined_loader import TOX_TASKS
from pipeline.preprocess import summarize_labels, to_mol
from pipeline.toxicity_loader import DATASETS

N_ESTIMATORS = 500
MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.1, 0.3]


def positive_probabilities(model, X):
    if len(X) == 0:
        return np.empty(0)
    positive_column = np.flatnonzero(model.classes_ == 1)[0]
    return model.predict_proba(X)[:, positive_column]


def fit_endpoint(X, y, train_idx, val_idx, seed=None):
    if seed is None:
        seed = SEEDS[0]
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
            random_state=seed,
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


def _lookup_auc(summary, endpoint, split):
    if summary.empty:
        return float("nan")
    match = summary[
        (summary["endpoint"] == endpoint)
        & (summary["split"] == split)
        & (summary["metric"] == "roc_auc")
    ]
    if match.empty:
        return float("nan")
    return float(match["value"].iloc[0])


def run_split(split_id):
    print(f"\n=== split {split_id} ===")
    df = load_benchmark_split(split_id=split_id)
    tox_df = df.loc[df[TOX_TASKS].notna().any(axis=1)].reset_index(drop=True)
    X = (
        np.stack([mol_to_ecfp4(to_mol(s), FINGERPRINT_BITS) for s in tox_df["smiles"]])
        if len(tox_df)
        else np.empty((0, FINGERPRINT_BITS), dtype=np.uint8)
    )
    y = tox_df[TOX_TASKS].to_numpy(dtype=float)
    observed = tox_df[TOX_TASKS].notna().to_numpy()
    split = tox_df["split"].to_numpy()
    indices = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}

    for name, idx in indices.items():
        print(f"\n{name}: {len(idx)} molecules with at least one toxicity label")
        print(summarize_labels(tox_df.iloc[idx], TOX_TASKS).round(4).to_string())

    selected = {}
    selections = []
    for j, task in enumerate(TOX_TASKS):
        train_idx, val_idx = (
            indices[name][observed[indices[name], j]] for name in ("train", "val")
        )
        model, status = fit_endpoint(X, y[:, j], train_idx, val_idx, seed=SEEDS[0])
        choice = model.max_features if model is not None else None
        selections.append((task, choice, status))
        if choice is not None:
            selected[task] = choice

    split_masks = {
        name: {
            task: df.loc[df["split"] == name, task].notna().to_numpy()
            for task in selected
        }
        for name in ("val", "test")
    }

    def fit_predict(split_df, seed):
        predictions = {"val": {}, "test": {}}
        for j, task in enumerate(TOX_TASKS):
            if task not in selected:
                continue
            train_idx = indices["train"][observed[indices["train"], j]]
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_features=selected[task],
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(X[train_idx], y[train_idx, j])
            for name in ("val", "test"):
                eval_idx = indices[name][observed[indices[name], j]]
                mask = split_masks[name][task]
                scores = np.full(len(mask), np.nan, dtype=float)
                scores[mask] = positive_probabilities(model, X[eval_idx])
                predictions[name][task] = scores
        return predictions

    params = {"max_features": selected, "n_estimators": N_ESTIMATORS}
    run_dir = benchmark.RUNS_DIR / f"rf_toxicity_{split_id}"
    run_benchmark(
        fit_predict,
        model="rf_toxicity",
        encoder=f"ecfp4-{FINGERPRINT_BITS}",
        params=params,
        seeds=SEEDS,
        split_id=split_id,
        df=df,
        run_dir=run_dir,
        overwrite_predictions=True,
    )
    scored = score_run_predictions(run_dir, df, include_test=True, n_boot=0)
    promote_run(
        run_dir,
        promoted_path=benchmark.PROMOTED_PATH,
        runs_dir=benchmark.RUNS_DIR,
        rebuild=False,
    )
    summary = summarize_seeds(scored)

    groups = {"overall": list(TOX_TASKS)}
    groups.update({dataset: list(cfg["tasks"]) for dataset, cfg in DATASETS.items()})
    for name in ("val", "test"):
        for group, tasks in groups.items():
            macro_auc = _lookup_auc(summary, f"macro_{group}", name)
            task_aucs = [_lookup_auc(summary, t, name) for t in tasks]
            count = int(np.isfinite(task_aucs).sum())
            print(
                f"{name} {group}: macro ROC-AUC={macro_auc:.4f} "
                f"({count}/{len(tasks)} evaluable endpoints)"
            )

    print("\nEndpoint results (FDA_APPROVED predicts approval status):")
    for task, max_features, status in selections:
        val_auc = _lookup_auc(summary, task, "val")
        test_auc = _lookup_auc(summary, task, "test")
        print(
            f"{task}: max_features={max_features}, "
            f"val ROC-AUC={val_auc:.4f}, "
            f"test ROC-AUC={test_auc:.4f}; {status}"
        )


def main():
    for split_id in SPLIT_IDS:
        run_split(split_id)
    missing = missing_promoted_runs(benchmark.PROMOTED_PATH, benchmark.RUNS_DIR)
    if not missing:
        rebuild_tracked_results(
            promoted_path=benchmark.PROMOTED_PATH,
            runs_dir=benchmark.RUNS_DIR,
        )
    else:
        print(
            f"Skipped rebuilding data/baseline/*.csv: missing promoted runs {missing} "
            "in runs/ (run `python drivers/download_mufu.py`, then "
            "`python drivers/rebuild_results.py`)."
        )


if __name__ == "__main__":
    main()
