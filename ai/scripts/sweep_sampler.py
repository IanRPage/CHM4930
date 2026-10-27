"""
A/B MuFu's MixedBatchSampler and AlternatingBatchSampler on the shared RF split.

Usage (from ai/):
    python scripts/sweep_sampler.py all
    python scripts/sweep_sampler.py all --force
    python scripts/sweep_sampler.py summary

Completed seeds are skipped unless --force is given. Commit settings before a
final run so the saved checkpoints and metrics record the experiment's commit.
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from baseline.benchmark import SEEDS, load_benchmark_split, run_benchmark
from baseline.results import RESULTS_PATH, summarize_seeds
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import MuFuDataset, split_datasets
from mufu.experiment import (
    DEFAULT_CONFIG,
    ENCODER,
    benchmark_params,
    make_fit_predict,
)
from mufu.train import TrainConfig

AI_DIR = Path(__file__).resolve().parents[1]
SWEEP_DIR = AI_DIR / "outputs" / "sampler"
SWEEP_RESULTS_PATH = SWEEP_DIR / "sweep_results.csv"
SWEEP_CKPT_DIR = AI_DIR / "checkpoints" / "sampler"
SWEEP_SEEDS: tuple[int, ...] = SEEDS
SAMPLERS = ("mixed", "alternating")
MODEL_NAMES = tuple(f"mufu_sampler_{sampler}" for sampler in SAMPLERS)


def get_completed_seeds(
    model_name: str, results_path: Path = SWEEP_RESULTS_PATH
) -> set[int]:
    if not results_path.exists():
        return set()
    rows = pd.read_csv(results_path)
    rows = rows[rows["model"] == model_name]
    return set(rows["seed"].dropna().astype(int).unique())


def run_variant(
    model_name: str,
    cfg: TrainConfig,
    df: pd.DataFrame,
    datasets: dict[str, MuFuDataset],
    smiles_encoder: SmilesEncoder,
    *,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
    results_path: Path = SWEEP_RESULTS_PATH,
    checkpoint_dir: Path = SWEEP_CKPT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    done = set() if force else get_completed_seeds(model_name, results_path)
    remaining = tuple(seed for seed in seeds if seed not in done)
    if not remaining:
        print(f"[skip] {model_name}: seeds {seeds} already recorded", flush=True)
        all_rows = pd.read_csv(results_path)
        return summarize_seeds(all_rows[all_rows["model"] == model_name])

    print(f"\n=== {model_name}: seeds {remaining} ===\n{cfg}", flush=True)
    started = time.monotonic()
    fit_predict = make_fit_predict(
        datasets,
        smiles_encoder,
        cfg,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        history_dir=SWEEP_DIR / model_name,
        verbose=True,
    )
    summary = run_benchmark(
        fit_predict,
        model=model_name,
        encoder=ENCODER,
        params=benchmark_params(cfg, smiles_encoder),
        seeds=remaining,
        df=df,
        path=results_path,
    )
    print(
        f"=== Finished {model_name} in {time.monotonic() - started:.1f}s ===",
        flush=True,
    )
    return summary


def print_sweep_summary(results_path: Path = SWEEP_RESULTS_PATH) -> pd.DataFrame:
    if not results_path.exists():
        print(f"No sweep results found at {results_path}")
        return pd.DataFrame()

    df = pd.read_csv(results_path)
    if RESULTS_PATH.exists():
        rf_df = pd.read_csv(RESULTS_PATH)
        snapshots = df["data_sha256"].dropna().unique()
        if len(snapshots) != 1:
            raise ValueError("sampler results must contain exactly one data_sha256")
        snapshot = snapshots[0]
        rf_df = rf_df[rf_df["model"].isin(["rf_bioactivity", "rf_toxicity"])]
        rf_df = rf_df[rf_df["data_sha256"] == snapshot]
        df = pd.concat([df, rf_df], ignore_index=True)

    summary = summarize_seeds(df)
    target_rows = [
        ("bace1", "pIC50_BACE1", "rmse"),
        ("bace1", "pIC50_BACE1", "r2"),
        ("bace1", "active_BACE1", "roc_auc"),
        ("egfr", "pIC50_EGFR", "rmse"),
        ("egfr", "pIC50_EGFR", "r2"),
        ("egfr", "active_EGFR", "roc_auc"),
        ("tox21", "macro_tox21", "roc_auc"),
        ("clintox", "macro_clintox", "roc_auc"),
        ("overall", "macro_overall", "roc_auc"),
    ]
    for sp in ("val", "test"):
        print(f"\n================ {sp.upper()} SPLIT ================")
        sub = summary[summary["split"] == sp]
        rows = []
        for model_name in sub["model"].unique():
            m_sub = sub[sub["model"] == model_name]
            entry: dict[str, object] = {
                "model": model_name,
                "n_seeds": int(m_sub["n_seeds"].max()),
            }
            for ds, ep, met in target_rows:
                match = m_sub[
                    (m_sub["dataset"] == ds)
                    & (m_sub["endpoint"] == ep)
                    & (m_sub["metric"] == met)
                ]
                if not match.empty:
                    v = float(match["value"].iloc[0])
                    s = float(match["std"].iloc[0])
                    entry[f"{ep}_{met}"] = f"{v:.3f}±{s:.3f}"
            rows.append(entry)
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "summary"), nargs="?", default="all")
    parser.add_argument("--force", action="store_true", help="rerun completed seeds")
    args = parser.parse_args(argv)
    if args.command == "summary":
        print_sweep_summary()
        return

    print("Loading and featurizing the shared benchmark split...", flush=True)
    df = load_benchmark_split()
    datasets = split_datasets(df)
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)} molecules", flush=True)

    # Reuse the frozen encoder and embedding cache across both variants and seeds.
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))
    base = replace(
        DEFAULT_CONFIG, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    configs = [replace(base, sampler=sampler) for sampler in SAMPLERS]
    for model_name, cfg in zip(MODEL_NAMES, configs):
        run_variant(model_name, cfg, df, datasets, smiles_encoder, force=args.force)
    print_sweep_summary()


if __name__ == "__main__":
    main()
