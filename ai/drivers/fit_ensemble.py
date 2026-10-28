"""
Fit the MuFu seed ensemble from saved checkpoints.

From ai/: python drivers/fit_ensemble.py
Settings are below; train_mufu.py calls `fit_and_report` after training.

Fits q on val, saves it as checkpoints/<model_name>_ensemble.json, and writes ensemble
metrics and interval coverage to outputs/<model_name>/. Nothing is written to
results.csv. Skipped if any seed checkpoint is missing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from baseline.benchmark import SEEDS, evaluate_split_predictions, load_benchmark_split
from baseline.results import RESULTS_PATH, summarize_seeds
from encoders.smiles_encoder import SmilesEncoder
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import split_datasets
from mufu.ensemble import (
    ensemble_predict,
    fit_ensemble_conformal,
    load_members,
    member_paths,
    predict_with_ensemble_intervals,
    save_ensemble,
)
from mufu.mc_dropout import DEFAULT_COVERAGES, DEFAULT_T, empirical_coverage

AI_DIR = Path(__file__).resolve().parents[1]

model_name = "mufu"
seeds = SEEDS
n_samples = DEFAULT_T
coverages = DEFAULT_COVERAGES
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# (dataset, endpoint, metric)
MAIN_METRICS = (
    ("bace1", "pIC50_BACE1", "rmse"),
    ("bace1", "pIC50_BACE1", "r2"),
    ("egfr", "pIC50_EGFR", "rmse"),
    ("egfr", "pIC50_EGFR", "r2"),
    ("tox21", "macro_tox21", "roc_auc"),
    ("clintox", "macro_clintox", "roc_auc"),
)
PREDICTION_COLUMNS = ("smiles", *PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS)


def single_model_summary(model_name: str) -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    recorded = pd.read_csv(RESULTS_PATH)
    return summarize_seeds(recorded[recorded["model"] == model_name])


def print_comparison(metrics: pd.DataFrame, single: pd.DataFrame) -> None:
    keys = ["dataset", "endpoint", "split", "metric"]
    merged = metrics.merge(single, on=keys, suffixes=("_ens", "_single"))
    merged = merged.set_index(keys)
    for split in ("test", "val"):
        print(f"\n{split} split: ensemble vs. single-model mean +/- std")
        for dataset, endpoint, metric in MAIN_METRICS:
            if (dataset, endpoint, split, metric) not in merged.index:
                continue
            row = merged.loc[(dataset, endpoint, split, metric)]
            print(
                f"{split} {dataset} {metric}: ensemble {row['value_ens']:.3f} | "
                f"single {row['value_single']:.3f} +/- {row['std']:.3f} "
                f"(n={int(row['n_seeds'])})"
            )


def fit_and_report(
    df: pd.DataFrame,
    datasets,
    smiles_encoder,
    *,
    model_name: str = model_name,
    seeds=seeds,
    n_samples: int = n_samples,
    coverages: tuple[float, ...] = coverages,
    batch_size: int = batch_size,
    device: str = device,
) -> bool:
    checkpoint_dir = AI_DIR / "checkpoints"
    try:
        paths = member_paths(checkpoint_dir, model_name, seeds)
    except FileNotFoundError as e:
        print(f"Ensemble skipped: {e}", flush=True)
        return False

    torch.manual_seed(0)
    members = load_members(paths, smiles_encoder, device=device)
    params = fit_ensemble_conformal(
        members,
        datasets["val"],
        n_samples=n_samples,
        coverages=coverages,
        batch_size=batch_size,
    )
    save_ensemble(params, checkpoint_dir, model_name)
    conformal = params.conformal
    print(f"ensemble q={conformal.q.round(3).tolist()}", flush=True)

    val_preds, _ = ensemble_predict(
        members, datasets["val"], n_samples=n_samples, batch_size=batch_size
    )
    test_preds = predict_with_ensemble_intervals(
        members, datasets["test"], params, batch_size=batch_size
    )
    metrics = pd.DataFrame(
        evaluate_split_predictions(
            df,
            {
                "val": val_preds,
                "test": {c: test_preds[c] for c in PREDICTION_COLUMNS},
            },
        )
    )
    coverage = empirical_coverage(test_preds, datasets["test"], conformal.coverages)

    out_dir = AI_DIR / "outputs" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "ensemble_metrics.csv", index=False)
    coverage.to_csv(out_dir / "ensemble_coverage.csv", index=False)

    single = single_model_summary(model_name)
    if not single.empty:
        print_comparison(metrics, single)
    print("\ntest split: combined-interval coverage")
    print(coverage.round(4).to_string(index=False))
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    df = load_benchmark_split()
    fit_and_report(
        df,
        split_datasets(df),
        SmilesEncoder(freeze=True, max_cache_size=len(df)),
    )
