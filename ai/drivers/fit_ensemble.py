"""
Fit (or load) the MuFu seed ensemble from saved checkpoints in `ai/runs/<run_id>/`.

From ai/: python drivers/fit_ensemble.py
Settings are below; `train_mufu.py` calls `fit_and_report` after training.

If `checkpoints/<model_name>_ensemble.json` already exists and `refit = False`, loads it
without refitting (preserving its exact bytes and sha256); otherwise fits `q` on `val`
and saves `<model_name>_ensemble.json`. Writes `predictions/test_ensemble.csv`. If the
run is promoted in `promoted.json`, also refreshes `ensemble_metrics.csv` and
`ensemble_coverage.csv`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from baseline.benchmark import (
    PROMOTED_PATH,
    RUNS_DIR,
    SEEDS,
    load_benchmark_split,
    save_split_predictions,
)
from baseline.results import DEFAULT_SPLIT_ID, load_results, summarize_seeds
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.ensemble import (
    fit_ensemble_conformal,
    load_ensemble_params,
    load_members,
    member_paths,
    predict_with_ensemble_intervals,
    save_ensemble,
    sidecar_path,
)
from mufu.experiment import score_mufu_run_test
from mufu.mc_dropout import DEFAULT_COVERAGES, DEFAULT_T

run_id = ""  # empty -> promoted["mufu"]["default"] from data/baseline/promoted.json
model_name = "mufu"
seeds = SEEDS
n_samples = DEFAULT_T
coverages = DEFAULT_COVERAGES
batch_size = 64
refit = False
overwrite_predictions = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# (dataset, endpoint, metric)
MAIN_METRICS = (
    ("bace1", "pIC50_BACE1", "rmse"),
    ("bace1", "pIC50_BACE1", "r2"),
    ("egfr", "pIC50_EGFR", "rmse"),
    ("egfr", "pIC50_EGFR", "r2"),
    ("tox21", "macro_tox21", "roc_auc"),
    ("clintox", "CT_TOX", "roc_auc"),
    ("clintox", "FDA_APPROVED", "roc_auc"),
)


def resolve_run_dir(explicit_run_id: str = "") -> Path:
    if explicit_run_id:
        return RUNS_DIR / explicit_run_id
    if PROMOTED_PATH.exists():
        promoted = json.loads(PROMOTED_PATH.read_text())
        default_id = promoted.get(model_name, {}).get(DEFAULT_SPLIT_ID)
        if default_id:
            return RUNS_DIR / default_id
    raise FileNotFoundError(
        "Set `run_id` at the top of the file or promote a default MuFu run first."
    )


def print_test_comparison(metrics: pd.DataFrame, single: pd.DataFrame) -> None:
    keys = ["dataset", "endpoint", "split", "metric"]
    merged = metrics.merge(single, on=keys, suffixes=("_ens", "_single"))
    merged = merged.set_index(keys)
    print("\ntest split: ensemble vs. single-model mean +/- std")
    for dataset, endpoint, metric in MAIN_METRICS:
        if (dataset, endpoint, "test", metric) not in merged.index:
            continue
        row = merged.loc[(dataset, endpoint, "test", metric)]
        print(
            f"test {dataset} {metric}: ensemble {row['value_ens']:.3f} | "
            f"single {row['value_single']:.3f} +/- {row['std']:.3f} "
            f"(n={int(row['n_seeds'])})"
        )


def fit_and_report(
    df: pd.DataFrame,
    datasets,
    smiles_encoder,
    *,
    run_dir: Path,
    model_name: str = model_name,
    seeds=seeds,
    n_samples: int = n_samples,
    coverages: tuple[float, ...] = coverages,
    batch_size: int = batch_size,
    device: str = device,
    refit: bool = False,
    overwrite_predictions: bool = False,
) -> bool:
    checkpoint_dir = run_dir / "checkpoints"
    try:
        paths = member_paths(checkpoint_dir, model_name, seeds)
    except FileNotFoundError as e:
        print(f"Ensemble skipped: {e}", flush=True)
        return False

    members = load_members(paths, smiles_encoder, device=device)
    ens_file = sidecar_path(checkpoint_dir, model_name)
    if ens_file.exists() and not refit:
        params = load_ensemble_params(paths, checkpoint_dir, model_name)
        print(
            f"loaded existing {ens_file.name}: q={params.conformal.q.round(3).tolist()}",
            flush=True,
        )
    else:
        params = fit_ensemble_conformal(
            members,
            datasets["val"],
            n_samples=n_samples,
            coverages=coverages,
            batch_size=batch_size,
        )
        save_ensemble(params, checkpoint_dir, model_name)
        print(f"ensemble q={params.conformal.q.round(3).tolist()}", flush=True)

    test_preds = predict_with_ensemble_intervals(
        members, datasets["test"], params, batch_size=batch_size
    )
    save_split_predictions(
        df,
        "test",
        test_preds,
        run_dir / "predictions" / "test_ensemble.csv",
        overwrite=overwrite_predictions,
    )

    is_promoted = False
    if PROMOTED_PATH.exists():
        promoted = json.loads(PROMOTED_PATH.read_text())
        is_promoted = promoted.get(model_name, {}).get(DEFAULT_SPLIT_ID) == run_dir.name

    if is_promoted:
        score_mufu_run_test(run_dir, df)
        metrics = pd.read_csv(run_dir / "ensemble_metrics.csv")
        coverage = pd.read_csv(run_dir / "ensemble_coverage.csv")
        recorded = load_results()
        single = summarize_seeds(
            recorded[
                (recorded["model"] == model_name)
                & (recorded["split_id"] == DEFAULT_SPLIT_ID)
            ]
        )
        if not single.empty:
            print_test_comparison(metrics, single)
        print("\ntest split: combined-interval coverage")
        print(coverage.round(4).to_string(index=False))

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    target_dir = resolve_run_dir(run_id)
    df = load_benchmark_split()
    fit_and_report(
        df,
        split_datasets(df),
        SmilesEncoder(freeze=True, max_cache_size=len(df)),
        run_dir=target_dir,
        refit=refit,
        overwrite_predictions=overwrite_predictions,
    )
