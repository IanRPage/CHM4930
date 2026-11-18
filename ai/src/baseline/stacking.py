"""Fit a weighted MuFu + RF blend on validation predictions and score it on test."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import benchmark as bm
from baseline.results import DEFAULT_SPLIT_ID, file_sha256, summarize_seeds
from evaluation import evaluation_metrics
from mufu.ensemble import average_predictions
from pipeline.combined_loader import COMBINED_CSV_PATH, TOX_TASKS
from pipeline.toxicity_loader import DATASETS

RUN_ENDPOINTS = {
    "mufu": (*bm.BIO_PIC50_COLS, *TOX_TASKS),
    "rf_bioactivity": bm.BIO_PIC50_COLS,
    "rf_toxicity": TOX_TASKS,
}
REPORT_METRICS = {
    **dict.fromkeys(bm.BIO_PIC50_COLS, ("rmse", "r2")),
    "macro_tox21": ("roc_auc",),
    **dict.fromkeys(DATASETS["clintox"]["tasks"], ("roc_auc",)),
    "macro_clintox": ("roc_auc",),
}
# Try a balanced blend first when several weights have the same validation AUC.
AUC_WEIGHTS = tuple(
    step / 100 for step in sorted(range(101), key=lambda s: abs(s - 50))
)


def _prediction_path(run_dir: Path, split: str, seed: int) -> Path:
    return run_dir / "predictions" / f"{split}_seed_{seed}.csv"


def _validate_runs(
    run_dirs: dict[str, Path], split_id: str, df, data_path: Path
) -> None:
    required = [
        path
        for run_dir in run_dirs.values()
        for path in (
            run_dir / "run.json",
            *(
                _prediction_path(run_dir, split, seed)
                for split in ("val", "test")
                for seed in bm.SEEDS
            ),
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark artifacts:\n" + "\n".join(missing))

    data_hash, split_hash = file_sha256(data_path), bm.split_sha256(df)
    for model, run_dir in run_dirs.items():
        meta, seeds = bm._load_run_meta(run_dir, require_complete=True, action="stack")
        expected = {
            "run_id": run_dir.name,
            "model": model,
            "split_id": split_id,
            "data_sha256": data_hash,
            "split_sha256": split_hash,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"{run_dir / 'run.json'}: {key} is {meta.get(key)!r}, expected {value!r}"
                )
        if not {"seeds", "completed_seeds"} <= meta.keys() or sorted(seeds) != list(
            bm.SEEDS
        ):
            raise ValueError(
                f"{run_dir / 'run.json'}: expected completed seeds {bm.SEEDS}"
            )


def _load_predictions(run_dir, model, split, seed, expected_smiles):
    path = _prediction_path(run_dir, split, seed)
    try:
        raw = bm.load_split_predictions(path)
    except KeyError as exc:
        raise ValueError(f"{path}: missing SMILES column") from exc
    smiles = pd.Index(raw["smiles"])
    if smiles.hasnans or smiles.has_duplicates:
        raise ValueError(f"{path}: missing or duplicate prediction SMILES")
    order = smiles.get_indexer(expected_smiles)
    if len(smiles) != len(expected_smiles) or (order < 0).any():
        missing = expected_smiles.difference(smiles, sort=False)
        extra = smiles.difference(expected_smiles, sort=False)
        raise ValueError(
            f"{path}: prediction SMILES do not match the {split} split "
            f"({len(missing)} missing, {len(extra)} unexpected)"
        )

    scores = {}
    for endpoint in RUN_ENDPOINTS[model]:
        if endpoint not in raw:
            continue
        values = raw[endpoint][order]
        if endpoint in TOX_TASKS:
            # MuFu's predictor emits the paired _uncal column when calibration
            # was applied. Read the calibrated scores from the unsuffixed column.
            if model == "mufu" and f"{endpoint}_uncal" not in raw:
                raise ValueError(f"{path}: missing calibration evidence for {endpoint}")
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{path}: {endpoint} probabilities must be in [0, 1]")
        scores[endpoint] = values
    return scores


def _load_pair(run_dirs, split_df, split, seed):
    smiles = pd.Index(split_df["smiles"])
    scores = {
        model: _load_predictions(run_dir, model, split, seed, smiles)
        for model, run_dir in run_dirs.items()
    }
    return scores["mufu"], scores["rf_bioactivity"] | scores["rf_toxicity"]


def _paired_rows(labels, mufu, rf):
    return np.isfinite(labels) & np.isfinite(mufu) & np.isfinite(rf)


def _blend(mufu, rf, weight):
    return weight * mufu + (1.0 - weight) * rf


def _fit_weight(labels, mufu, rf, *, classification):
    if classification:

        def auc(weight):
            return evaluation_metrics(
                labels, _blend(mufu, rf, weight), "classification"
            )["roc_auc"]

        return max(AUC_WEIGHTS, key=auc)

    # Minimizing RMSE is equivalent to minimizing the squared residuals.
    delta = mufu - rf
    scale = delta @ delta
    return float(np.clip(delta @ (labels - rf) / scale, 0, 1)) if scale else 0.5


def _fit_validation(val_df, mufu, rf):
    weights = {}
    for endpoint in RUN_ENDPOINTS["mufu"]:
        labels = val_df[endpoint].to_numpy(dtype=float)
        fit = {
            "mufu_weight": np.nan,
            "val_labeled": int(np.isfinite(labels).sum()),
            "val_paired": 0,
            "status": "fit",
        }
        weights[endpoint] = fit
        missing = [
            name for name, pred in (("MuFu", mufu), ("RF", rf)) if endpoint not in pred
        ]
        if missing:
            fit["status"] = f"unsupported by {' and '.join(missing)}"
            continue
        mask = _paired_rows(labels, mufu[endpoint], rf[endpoint])
        y = labels[mask]
        fit["val_paired"] = len(y)
        classification = endpoint in TOX_TASKS
        if not len(y):
            fit["status"] = "no paired validation labels"
        elif classification and np.unique(y).size < 2:
            fit["status"] = "single-class validation labels"
        else:
            fit["mufu_weight"] = _fit_weight(
                y,
                mufu[endpoint][mask],
                rf[endpoint][mask],
                classification=classification,
            )
    return weights


def _metric_rows(frame, scores, model):
    scored = {
        (row["endpoint"], row["metric"]): row["value"]
        for row in bm.evaluate_split_predictions(frame, {"test": scores})
    }
    return [
        {
            "model": model,
            "endpoint": endpoint,
            "metric": metric,
            "value": scored.get((endpoint, metric), np.nan),
        }
        for endpoint, metrics in REPORT_METRICS.items()
        for metric in metrics
    ]


def _ensemble_predictions(frame, pairs):
    # _load_pair has already joined each file to this frame's SMILES.
    members = []
    for seed in bm.SEEDS:
        scores, _ = pairs[seed]
        if scores.keys() != pairs[bm.SEEDS[0]][0].keys():
            raise ValueError(f"MuFu seed {seed}: ensemble endpoint mismatch")
        members.append({"smiles": frame["smiles"].tolist(), **scores})
    mean = average_predictions(members)
    del mean["smiles"]
    return mean


def _common_test_frame(test_df, test_pairs, fits, ensemble_mean):
    """Use one labeled cohort per endpoint for the fixed score and every seed."""
    paired_df = test_df.copy()
    for endpoint in RUN_ENDPOINTS["mufu"]:
        mask = np.isfinite(test_df[endpoint].to_numpy(dtype=float))
        for seed, (mu, rf) in test_pairs.items():
            if not np.isfinite(fits[seed][endpoint]["mufu_weight"]):
                # A five-seed comparison needs this endpoint in every pair.
                mask[:] = False
                break
            mask &= np.isfinite(mu[endpoint]) & np.isfinite(rf[endpoint])
        if ensemble_mean is not None and endpoint in ensemble_mean:
            mask &= np.isfinite(ensemble_mean[endpoint])
        paired_df.loc[~mask, endpoint] = np.nan
    return paired_df


def _score_test(paired_df, mufu, rf, weights, *, ensemble=False):
    predictions = {name: {} for name in ("MuFu", "RF", "Blend")}
    paired_counts = dict.fromkeys(weights, 0)
    for endpoint, fit in weights.items():
        weight = fit["mufu_weight"]
        if np.isnan(weight):
            continue
        mu, forest = mufu[endpoint], rf[endpoint]
        mask = paired_df[endpoint].notna().to_numpy()
        paired_counts[endpoint] = int(mask.sum())
        predictions["MuFu"][endpoint] = mu
        predictions["RF"][endpoint] = forest
        blended = np.full(len(paired_df), np.nan)
        blended[mask] = _blend(mu[mask], forest[mask], weight)
        predictions["Blend"][endpoint] = blended

    if ensemble:
        predictions = {"Ensemble + RF": predictions["Blend"]}
    rows = [
        row
        for model, scores in predictions.items()
        for row in _metric_rows(paired_df, scores, model)
    ]
    return rows, paired_counts


def evaluate_stacking(
    split_id: str = DEFAULT_SPLIT_ID,
    *,
    runs_dir: Path | None = None,
    data_path: Path = COMBINED_CSV_PATH,
    ensemble: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return test seed summaries and per-seed validation fits for one scaffold split."""
    run_dirs = {
        model: bm.resolve_run_dir(model=model, split_id=split_id, runs_dir=runs_dir)
        for model in RUN_ENDPOINTS
    }
    df = bm.load_benchmark_split(data_path, split_id)
    smiles = pd.Index(df["smiles"])
    if smiles.hasnans or smiles.has_duplicates:
        raise ValueError("Benchmark snapshot contains missing or duplicate SMILES")
    _validate_runs(run_dirs, split_id, df, data_path)
    val_df, test_df = (df[df["split"] == split] for split in ("val", "test"))

    metric_rows, weight_rows = [], []
    val_pairs = {s: _load_pair(run_dirs, val_df, "val", s) for s in bm.SEEDS}
    val_mean = _ensemble_predictions(val_df, val_pairs) if ensemble else None
    fits = {
        seed: _fit_validation(val_df, val_mean if ensemble else mu, rf)
        for seed, (mu, rf) in val_pairs.items()
    }
    test_pairs = {s: _load_pair(run_dirs, test_df, "test", s) for s in bm.SEEDS}
    for seed, (test_mufu, test_rf) in test_pairs.items():
        val_mufu, val_rf = val_pairs[seed]
        for model, val, test in (
            ("MuFu", val_mufu, test_mufu),
            ("RF", val_rf, test_rf),
        ):
            mismatched = val.keys() ^ test.keys()
            if mismatched:
                raise ValueError(
                    f"{model} seed {seed}: validation/test endpoint mismatch {sorted(mismatched)}"
                )
    test_mean = _ensemble_predictions(test_df, test_pairs) if ensemble else None
    paired_df = _common_test_frame(test_df, test_pairs, fits, test_mean)
    if ensemble:
        metric_rows.extend(
            {**row, "split_id": split_id, "split": "test", "seed": None}
            for row in _metric_rows(paired_df, test_mean, "MuFu ensemble")
        )
    for seed, (test_mufu, test_rf) in test_pairs.items():
        weights = fits[seed]
        rows, test_counts = _score_test(
            paired_df,
            test_mean if ensemble else test_mufu,
            test_rf,
            weights,
            ensemble=ensemble,
        )
        metric_rows.extend(
            {**row, "split_id": split_id, "seed": seed, "split": "test"} for row in rows
        )
        weight_rows.extend(
            {
                "split_id": split_id,
                "seed": seed,
                "endpoint": endpoint,
                **fit,
                "test_paired": test_counts[endpoint],
            }
            for endpoint, fit in weights.items()
        )

    columns = ["model", "endpoint", "metric", "value", "std", "n_seeds"]
    summary = summarize_seeds(pd.DataFrame(metric_rows))[columns].rename(
        columns={"value": "mean"}
    )
    if ensemble:
        summary = summary.rename(columns={"n_seeds": "n_evaluations"})
    return summary, pd.DataFrame(weight_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Compare the fixed five-member MuFu ensemble with its RF blends",
    )
    parser.add_argument("--split-id", choices=bm.SPLIT_IDS, default=DEFAULT_SPLIT_ID)
    parser.add_argument(
        "--runs-dir", type=Path, help="Directory containing the saved runs"
    )
    args = parser.parse_args()
    try:
        summary, weights = evaluate_stacking(
            args.split_id, runs_dir=args.runs_dir, ensemble=args.ensemble
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Test metrics on {args.split_id}")
    if args.ensemble:
        print(
            "Fixed MuFu ensemble scored once. Blend mean and sample std across RF seeds and their validation fits."
        )
    else:
        print("Mean and sample std over evaluable seed pairs.")
    print(summary.round(4).to_string(index=False))
    print("\nValidation fits (weight on MuFu)")
    print(weights.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
