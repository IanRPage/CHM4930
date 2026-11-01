"""
Train MuFu on the shared benchmark seeds.

From ai/: python drivers/train_mufu.py
After the seeds finish, fits the seed ensemble (see fit_ensemble.py), skipping it if
any seed checkpoint is missing. Another `split_id` writes to checkpoints/<split_id>/
and outputs/mufu/<split_id>/, and skips the ensemble.

Settings below. Commit changes before a final run so results.csv gets a clean commit.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from baseline.benchmark import SEEDS, load_benchmark_split, run_benchmark
from encoders.smiles_encoder import SmilesEncoder
from fit_ensemble import fit_and_report
from mufu.data.dataset import split_datasets
from mufu.experiment import (
    DEFAULT_CONFIG,
    ENCODER,
    benchmark_params,
    make_fit_predict,
)

AI_DIR = Path(__file__).resolve().parents[1]

# training settings
model_name = "mufu"
seeds = SEEDS  # 0, 1, 2, 3, 4
split_id = "default"  # or "scaffold-0" ... "scaffold-4"
ensemble_after_training = True
# override fields here
config = replace(DEFAULT_CONFIG, device="cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Set training options at the top of this file, not as arguments."
        )

    run_dir = "" if split_id == "default" else split_id
    print("Loading and featurizing the shared split...", flush=True)
    df = load_benchmark_split(split_id=split_id)
    datasets = split_datasets(df)
    for split in datasets:
        print(f"{split}: {len(datasets[split])} molecules")
    print(config, flush=True)

    # one frozen smiles encoder for each seed so embedding cache only filled once
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))
    fit_predict = make_fit_predict(
        datasets,
        smiles_encoder,
        config,
        model_name=model_name,
        checkpoint_dir=AI_DIR / "checkpoints" / run_dir,
        history_dir=AI_DIR / "outputs" / model_name / run_dir,
        verbose=True,
    )
    summary = run_benchmark(
        fit_predict,
        model=model_name,
        encoder=ENCODER,
        params=benchmark_params(config, smiles_encoder),
        seeds=seeds,
        split_id=split_id,
        df=df,
    )
    columns = ["dataset", "endpoint", "split", "metric", "value", "std", "n_seeds"]
    print(summary[columns].round(4).to_string(index=False))

    if ensemble_after_training and not run_dir:
        fit_and_report(df, datasets, smiles_encoder, model_name=model_name, seeds=SEEDS)
