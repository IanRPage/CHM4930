"""Compare MuFu training with scaffold and UMAP splits on one shared test set.

From ai/:
    python scripts/sweep_split_methods.py prepare
    python scripts/sweep_split_methods.py all
    python scripts/sweep_split_methods.py all --force
    python scripts/sweep_split_methods.py rf --output-dir outputs/split_methods_leaf
    python scripts/sweep_split_methods.py summary
    python scripts/sweep_split_methods.py all --seeds 0 --epochs 3 --output-dir outputs/split_smoke

UMAP uses its defaults (80/10/10, seed 42). Its test molecules are shared by both
methods and all training seeds; scaffold splitting divides the remaining pool
into train/validation at 8:1. Assignments are saved and reused. MuFu uses
DEFAULT_CONFIG, including validation-based early stopping and calibration.
RF uses both saved allocations, tunes max_features separately on each allocation's
validation set with seed 0, and fits one 500-tree regressor per target and seed.
`all` runs all four model/allocation combinations; `rf` adds missing RF runs
without loading MuFu. Existing RF UMAP results are reused.

Reports contain source/label coverage, scaffold overlap, standard metrics, and
paired test RMSE differences (UMAP minus scaffold; negative favors UMAP).
Validation sets and training sizes can differ. The shared holdout may overlap
training scaffolds. --tolerance labels numerical differences, not significance.

Completed seeds are skipped; --force preserves old run artifacts and replaces
their summary rows. Changed data, assignments, model settings, or model code
require a new output directory.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from baseline import baseline_bioactivity as rf
from baseline.benchmark import (
    SEEDS,
    run_benchmark,
    score_run_predictions,
    split_sha256,
)
from baseline.results import code_identity, file_sha256, params_json, summarize_seeds
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.experiment import DEFAULT_CONFIG, ENCODER, benchmark_params, make_fit_predict
from pipeline.combined_loader import (
    COMBINED_CSV_PATH,
    TARGETS,
    TOX_TASKS,
    load_combined,
)
from pipeline.split import (
    SPLITS,
    murcko_scaffold,
    scaffold_split,
    split_summary,
    umap_split,
)

AI_DIR = Path(__file__).resolve().parents[1]
SWEEP_DIR = AI_DIR / "outputs" / "split_methods"
MODEL_NAMES = ("mufu_scaffold", "mufu_umap")
RF_MODEL = "rf_umap"
RF_MODELS = ("rf_scaffold", RF_MODEL)
METHODS = ("scaffold", "umap")


def shared_test_hash(df: pd.DataFrame) -> str:
    """Include test row order, which must also match when scoring predictions."""
    smiles = df.loc[df["split"] == "test", "smiles"].tolist()
    return hashlib.sha256(json.dumps(smiles).encode()).hexdigest()


def validate_splits(frames: dict[str, pd.DataFrame]) -> None:
    reference = frames["umap"]
    for method, frame in frames.items():
        if not frame["smiles"].equals(reference["smiles"]):
            raise ValueError(f"{method}: molecule rows/order differ from UMAP")
        if not frame["split"].isin(SPLITS).all():
            raise ValueError(f"{method}: unrecognized or missing split assignments")
        if set(frame["split"]) != set(SPLITS):
            raise ValueError(f"{method}: train, validation, and test must be nonempty")
        if not frame["split"].eq("test").equals(reference["split"].eq("test")):
            raise ValueError(
                f"{method}: test molecules differ from the frozen UMAP test"
            )


def build_splits(df: pd.DataFrame):
    """Use the existing splitters without changing pipeline/split.py."""
    if df["smiles"].isna().any() or df["smiles"].duplicated().any():
        raise ValueError("The experiment requires unique, nonmissing SMILES rows")
    df = df.reset_index(drop=True)
    umap = umap_split(df)
    test = umap["split"].eq("test")
    development = df.loc[~test].copy()
    # A two-way call: the splitter's local 'test' bucket is our validation pool.
    # The fixed UMAP test rows are excluded; 80:10 becomes 8:1 on this pool.
    development = scaffold_split(
        development,
        frac=(8 / 9, 0.0, 1 / 9),
        seed=42,
    )
    scaffold = df.assign(group="", split="test")
    scaffold.loc[development.index, "group"] = development["group"]
    scaffold.loc[development.index, "split"] = development["split"].replace(
        {"test": "val"}
    )
    scaffold.loc[test, "group"] = df.loc[test, "smiles"].map(murcko_scaffold)
    frames = {"scaffold": scaffold, "umap": umap}
    validate_splits(frames)
    return frames


def write_coverage(frames, output_dir: Path) -> None:
    sources, labels, overlaps = [], [], []
    label_cols = [
        *(f"pIC50_{t}" for t in TARGETS),
        *(f"active_{t}" for t in TARGETS),
        *TOX_TASKS,
    ]
    scaffolds = frames["scaffold"]["group"]
    for method, frame in frames.items():
        source = split_summary(frame).rename_axis("source").reset_index()
        sources.append(source.assign(method=method))
        for split in SPLITS:
            subset = frame.loc[frame["split"] == split]
            for column in label_cols:
                if column not in subset:
                    continue
                observed = subset[column].dropna()
                binary = not column.startswith("pIC50_")
                labels.append(
                    {
                        "method": method,
                        "split": split,
                        "endpoint": column,
                        "n_rows": len(subset),
                        "n_labeled": len(observed),
                        "n_negative": int(observed.eq(0).sum()) if binary else np.nan,
                        "n_positive": int(observed.eq(1).sum()) if binary else np.nan,
                    }
                )
        train_scaffolds = set(scaffolds[frame["split"].eq("train")]) - {""}
        test_scaffolds = scaffolds[frame["split"].eq("test")]
        overlaps.append(
            {
                "method": method,
                "n_test": len(test_scaffolds),
                "n_test_with_rings": int(test_scaffolds.ne("").sum()),
                "n_test_with_training_scaffold": int(
                    test_scaffolds.isin(train_scaffolds).sum()
                ),
            }
        )
    pd.concat(sources, ignore_index=True).to_csv(
        output_dir / "source_coverage.csv", index=False
    )
    pd.DataFrame(labels).to_csv(output_dir / "label_coverage.csv", index=False)
    pd.DataFrame(overlaps).to_csv(output_dir / "scaffold_overlap.csv", index=False)


def prepare_splits(output_dir: Path):
    """Save both allocations once, then reuse the same molecule assignments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "split_assignments.csv"
    df = load_combined(path=COMBINED_CSV_PATH).reset_index(drop=True)
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path, keep_default_na=False)
        if assignments["smiles"].tolist() != df["smiles"].tolist():
            raise ValueError("Saved split molecule rows/order do not match the dataset")
        frames = {
            method: df.assign(
                group=assignments[f"{method}_group"],
                split=assignments[f"{method}_split"],
            )
            for method in METHODS
        }
    else:
        if (output_dir / "sweep_results.csv").exists():
            raise ValueError(
                "Results exist without saved assignments; use a new --output-dir"
            )
        print(
            "Preparing UMAP test holdout and both train/validation allocations...",
            flush=True,
        )
        frames = build_splits(df)
        assignments = df[["smiles"]].copy()
        for method, frame in frames.items():
            assignments[f"{method}_group"] = frame["group"]
            assignments[f"{method}_split"] = frame["split"]
        assignments.to_csv(assignments_path, index=False, lineterminator="\n")
    validate_splits(frames)
    write_coverage(frames, output_dir)
    for method, frame in frames.items():
        print(
            f"\n{method} split sizes/source coverage:\n{split_summary(frame)}",
            flush=True,
        )
    return frames


def completed_seeds(output_dir, model_name, params, frame, identity):
    path = output_dir / "sweep_results.csv"
    if not path.exists():
        return set()
    rows = pd.read_csv(path)
    rows = rows.loc[rows["model"].eq(model_name)]
    expected = {
        "data_sha256": file_sha256(COMBINED_CSV_PATH),
        "params": params_json(params),
        "ai_tree": identity.ai_tree,
        "split_sha256": split_sha256(frame),
        "shared_test_sha256": shared_test_hash(frame),
    }
    for column, value in expected.items():
        if not rows[column].eq(value).all():
            raise ValueError(f"{model_name}: {column} changed; use a new --output-dir")
    return {
        int(seed)
        for seed, group in rows.groupby("seed")
        if set(group["split"]) == {"val", "test"}
    }


def compare_results(
    results: pd.DataFrame,
    tolerance: float,
    baseline_model: str = MODEL_NAMES[0],
    candidate_model: str = MODEL_NAMES[1],
) -> pd.DataFrame:
    """Pair test RMSE across allocations or model families on matching rows/seeds."""
    keys = ["dataset", "endpoint", "split", "metric"]
    pair_keys = ["seed", "data_sha256", "shared_test_sha256", *keys]
    cross_model = baseline_model.split("_", 1)[0] != candidate_model.split("_", 1)[0]
    # Across model families, require the entire partition to match. Across
    # allocations, training/validation differ while the test set stays fixed.
    pair_keys.append("split_sha256" if cross_model else "encoder")
    reference = "rf" if cross_model else "scaffold"
    rmse = results.loc[results["metric"].eq("rmse") & results["split"].eq("test")]
    baseline = rmse.loc[rmse["model"].eq(baseline_model), [*pair_keys, "value"]]
    candidate = rmse.loc[rmse["model"].eq(candidate_model)]
    paired = candidate.merge(
        baseline.rename(columns={"value": "baseline_rmse"}),
        on=pair_keys,
        how="inner",
        validate="one_to_one",
    )
    paired["delta"] = paired["value"] - paired["baseline_rmse"]
    comparison = paired.groupby(keys, as_index=False).agg(
        delta_vs_baseline=("delta", "mean"),
        delta_std=("delta", "std"),
        n_pairs=("delta", "count"),
    )
    delta = comparison.pop("delta_vs_baseline")
    comparison[f"delta_vs_{reference}"] = delta
    comparison[f"vs_{reference}"] = np.select(
        [delta.isna(), delta < -tolerance, delta > tolerance],
        ["unavailable", "improves", "worsens"],
        default="same",
    )
    return comparison


def print_sweep_summary(output_dir: Path, tolerance: float = 0.001):
    path = output_dir / "sweep_results.csv"
    if not path.exists():
        print(f"No completed sweep results found at {path}")
        return
    results = pd.read_csv(path)
    results = results.loc[results["model"].isin((*MODEL_NAMES, *RF_MODELS))]
    if results.empty:
        print("No completed split comparison runs found.")
        return
    for column in ("data_sha256", "shared_test_sha256", "ai_tree"):
        if results[column].nunique(dropna=False) != 1:
            raise ValueError(f"Sweep results must have one consistent {column}")
    for column in ("split_sha256", "params", "encoder"):
        if results.groupby("model")[column].nunique().gt(1).any():
            raise ValueError(f"Each model must have exactly one {column}")
    if results.loc[results["model"].isin(MODEL_NAMES), "params"].nunique() > 1:
        raise ValueError(
            "Both MuFu splitting methods must use the same model parameters"
        )
    metrics = summarize_seeds(results)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics.loc[
        metrics["split"].eq("test")
        & metrics["metric"].isin(("rmse", "r2", "spearman", "kendall"))
    ].to_csv(output_dir / "regression_summary.csv", index=False)
    for metric, label in (("rmse", "RMSE"), ("r2", "R²")):
        final = metrics.loc[metrics["metric"].eq(metric)]
        final.to_csv(output_dir / f"{metric}_summary.csv", index=False)
        print(
            f"\nSelected-checkpoint {label} (validation sets differ; test set is shared):"
        )
        print(
            final[["model", "split", "endpoint", "value", "std", "n_seeds"]].to_string(
                index=False
            )
        )
    comparisons = (
        ("mufu_umap", "mufu_scaffold", "rmse_comparison.csv"),
        ("rf_umap", "rf_scaffold", "rf_split_comparison.csv"),
        ("mufu_umap", "rf_umap", "rf_comparison.csv"),
        ("mufu_scaffold", "rf_scaffold", "rf_scaffold_comparison.csv"),
    )
    available = set(results["model"])
    for candidate, baseline, filename in comparisons:
        if not {candidate, baseline}.issubset(available):
            continue
        comparison = compare_results(results, tolerance, baseline, candidate)
        comparison.to_csv(output_dir / filename, index=False)
        print(
            f"\nPaired test RMSE: {candidate} minus {baseline} "
            f"(negative favors {candidate}; tolerance={tolerance}):"
        )
        print(comparison.to_string(index=False))
    print(f"\nReports saved in {output_dir}")


def save_scored_run(run_dir: Path, frame: pd.DataFrame, output_dir: Path):
    """Record one completed model/seed, replacing that pair on forced reruns."""
    scored = score_run_predictions(
        run_dir, df=frame, include_test=True, n_boot=0, data_path=COMBINED_CSV_PATH
    ).assign(shared_test_sha256=shared_test_hash(frame))
    model_name, seed = scored[["model", "seed"]].iloc[0]
    path = output_dir / "sweep_results.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old.loc[~(old["model"].eq(model_name) & old["seed"].eq(seed))]
        scored = pd.concat([old, scored], ignore_index=True)
    temporary = path.with_suffix(".csv.tmp")
    scored.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def make_rf_predictor(frame: pd.DataFrame, params=None):
    """Reuse the existing baseline's featurization and validation-only tuning."""
    prepared = {target: rf.prepare_target(frame, target) for target in TARGETS}
    for target, data in prepared.items():
        if not len(data["train_idx"]) or not len(data["val_idx"]):
            raise ValueError(f"RF {target} needs labeled training and validation rows")
    if params is None:
        params = {
            "max_features": {
                target: rf.select_max_features(data, seed=SEEDS[0])
                for target, data in prepared.items()
            },
            "n_estimators": rf.N_ESTIMATORS,
        }

    def fit_predict(split_df, seed):
        predictions = {"val": {}, "test": {}}
        for target, data in prepared.items():
            model = rf.RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_features=params["max_features"][target],
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(data["X"][data["train_idx"]], data["y"][data["train_idx"]])
            for split, split_predictions in predictions.items():
                mask = data["split_masks"][split]
                scores = np.full(len(mask), np.nan)
                if mask.any():
                    scores[mask] = model.predict(data["X"][data[f"{split}_idx"]])
                split_predictions[f"pIC50_{target}"] = scores
        return predictions

    return fit_predict, params


def run_rf(args, frames: dict[str, pd.DataFrame], identity):
    validate_splits(frames)
    for method in METHODS:
        run_rf_split(args, frames[method], identity, method)


def run_rf_split(args, frame: pd.DataFrame, identity, method: str):
    model_name = f"rf_{method}"
    path = args.output_dir / "sweep_results.csv"
    params = None
    if path.exists():
        results = pd.read_csv(path)
        # Validate against the MuFu arm with this same allocation, then reuse
        # only this RF arm's previously selected hyperparameters.
        for model in (f"mufu_{method}", model_name):
            rows = results.loc[results["model"].eq(model)]
            if not rows.empty:
                saved_params = json.loads(rows["params"].iloc[0])
                completed_seeds(args.output_dir, model, saved_params, frame, identity)
                if model == model_name:
                    params = saved_params
    done = (
        completed_seeds(args.output_dir, model_name, params, frame, identity)
        if params
        else set()
    )
    seeds = tuple(dict.fromkeys(s for s in args.seeds if args.force or s not in done))
    if not seeds:
        print(f"[skip] {model_name}: selected seeds already recorded", flush=True)
        return
    print(f"\nPreparing RF baseline on the saved {method} allocation...", flush=True)
    fit_predict, params = make_rf_predictor(frame, params)
    for seed in seeds:
        run_dir = (
            args.output_dir / "runs" / f"{model_name}_seed_{seed}_{time.time_ns()}"
        )
        print(f"\n=== {model_name}: seed {seed} ===\n{params}", flush=True)
        run_benchmark(
            fit_predict,
            model=model_name,
            encoder=f"ecfp4-{rf.FINGERPRINT_BITS}",
            params=params,
            seeds=(seed,),
            split_id=f"shared-umap-42-{method}",
            df=frame,
            data_path=COMBINED_CSV_PATH,
            run_dir=run_dir,
            identity=identity,
        )
        save_scored_run(run_dir, frame, args.output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=("all", "prepare", "rf", "summary"), nargs="?", default="all"
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun completed training seeds"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), help="training seeds"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--output-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args(argv)
    if args.epochs < 1 or not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--epochs must be positive and --tolerance finite and nonnegative")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if args.command == "summary":
        print_sweep_summary(args.output_dir, args.tolerance)
        return

    frames = prepare_splits(args.output_dir)
    if args.command == "prepare":
        print(
            f"\nAssignments and coverage saved in {args.output_dir}; no training performed."
        )
        return

    identity = code_identity()
    if args.command == "rf":
        run_rf(args, frames, identity)
        print_sweep_summary(args.output_dir, args.tolerance)
        return
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(frames["umap"]))
    cfg = replace(
        DEFAULT_CONFIG,
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    params = benchmark_params(cfg, smiles_encoder)
    # Validate both methods before starting any new training.
    done = {
        model: completed_seeds(args.output_dir, model, params, frames[method], identity)
        for method, model in zip(METHODS, MODEL_NAMES)
    }
    for method, model_name in zip(METHODS, MODEL_NAMES):
        seeds = tuple(
            dict.fromkeys(
                s for s in args.seeds if args.force or s not in done[model_name]
            )
        )
        if not seeds:
            print(f"[skip] {model_name}: selected seeds already recorded", flush=True)
            continue
        frame = frames[method]
        print(f"\nFeaturizing {method} train/validation/test...", flush=True)
        datasets = split_datasets(frame)
        for split in SPLITS:
            if len(datasets[split]) != int(frame["split"].eq(split).sum()):
                raise ValueError(f"{method}/{split}: featurization dropped molecules")
        print(f"\n=== {model_name}: training seeds {seeds} ===\n{cfg}", flush=True)
        for seed in seeds:
            # A fresh attempt also preserves artifacts from interrupted/forced runs.
            run_dir = (
                args.output_dir / "runs" / f"{model_name}_seed_{seed}_{time.time_ns()}"
            )
            fit_predict = make_fit_predict(
                datasets,
                smiles_encoder,
                cfg,
                run_dir=run_dir,
                model_name=model_name,
                identity=identity,
                data_path=COMBINED_CSV_PATH,
                verbose=True,
            )
            started = time.monotonic()
            run_benchmark(
                fit_predict,
                model=model_name,
                encoder=ENCODER,
                params=params,
                seeds=(seed,),
                split_id=f"shared-umap-42-{method}",
                df=frame,
                data_path=COMBINED_CSV_PATH,
                run_dir=run_dir,
                identity=identity,
            )
            save_scored_run(run_dir, frame, args.output_dir)
            print(
                f"Finished {model_name} seed {seed} in {time.monotonic() - started:.1f}s",
                flush=True,
            )
        del fit_predict, datasets
    run_rf(args, frames, identity)
    print_sweep_summary(args.output_dir, args.tolerance)


if __name__ == "__main__":
    main()
