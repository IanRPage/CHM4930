"""
Fit MC-Dropout conformal intervals for saved MuFu checkpoints in `ai/runs/<run_id>/`.

From ai/: python drivers/fit_mc_dropout.py
Fits `q` on `val`, saves it next to each seed's checkpoint, and writes empirical test
coverage to `ai/runs/<run_id>/logs/mc_dropout_coverage.csv`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from baseline.benchmark import PROMOTED_PATH, RUNS_DIR, SEEDS, load_benchmark_split
from baseline.results import DEFAULT_SPLIT_ID
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.experiment import load_mufu_checkpoint
from mufu.mc_dropout import (
    DEFAULT_COVERAGES,
    DEFAULT_T,
    empirical_coverage,
    fit_mc_dropout,
    predict_with_intervals,
)

run_id = ""  # empty -> promoted["mufu"]["default"] from data/baseline/promoted.json
model_name = "mufu"
seeds = SEEDS
n_samples = DEFAULT_T
coverages = DEFAULT_COVERAGES
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    run_dir = resolve_run_dir(run_id)
    df = load_benchmark_split()
    datasets = split_datasets(df)
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in seeds:
        ckpt_path = run_dir / "checkpoints" / f"{model_name}_seed_{seed}.pt"
        model, _ = load_mufu_checkpoint(ckpt_path, smiles_encoder, device=device)
        conformal = fit_mc_dropout(
            model,
            datasets["val"],
            ckpt_path,
            n_samples=n_samples,
            coverages=coverages,
            batch_size=batch_size,
        )
        preds = predict_with_intervals(
            model, datasets["test"], conformal, batch_size=batch_size
        )
        coverage = empirical_coverage(preds, datasets["test"], conformal.coverages)
        frames.append(coverage.assign(seed=seed))
        print(f"seed {seed}: q={conformal.q.round(3).tolist()}", flush=True)

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(
        logs_dir / "mc_dropout_coverage.csv", index=False, lineterminator="\n"
    )
    summary = result.groupby(["target", "nominal"])[["empirical", "mean_width"]].agg(
        ["mean", "std"]
    )
    print(summary.round(4).to_string())
