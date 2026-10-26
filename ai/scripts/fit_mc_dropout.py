"""
Fit MC-Dropout conformal intervals for saved MuFu checkpoints.

From ai/: python scripts/fit_mc_dropout.py

Fits q on val, saves it next to each seed's checkpoint, and reports empirical coverage
on test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from baseline.benchmark import SEEDS, load_benchmark_split
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

AI_DIR = Path(__file__).resolve().parents[1]

model_name = "mufu"
seeds = SEEDS
n_samples = DEFAULT_T
coverages = DEFAULT_COVERAGES
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    df = load_benchmark_split()
    datasets = split_datasets(df)
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))

    out_dir = AI_DIR / "outputs" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in seeds:
        ckpt_path = AI_DIR / "checkpoints" / f"{model_name}_seed_{seed}.pt"
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
    result.to_csv(out_dir / "mc_dropout_coverage.csv", index=False)
    summary = result.groupby(["target", "nominal"])[["empirical", "mean_width"]].agg(
        ["mean", "std"]
    )
    print(summary.round(4).to_string())
