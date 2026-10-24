"""
Train MuFu with the existing trainer and shared benchmark.

From ai/: python scripts/train_mufu.py
Edit the training settings below; the default runs all five shared seeds. Commit
setting changes before a final run so results.csv records a clean git commit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from baseline.benchmark import SEEDS, load_benchmark_split
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.experiment import benchmark_params, make_fit_predict, run_mufu_benchmark
from mufu.train import TrainConfig

AI_DIR = Path(__file__).resolve().parents[1]
ENCODER = "molformer-ecfp4-gine"

# Training settings
model_name = "mufu"
seeds = SEEDS  # Shared baseline seeds: 0, 1, 2, 3, 4.
config = TrainConfig(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=50,
    patience=10,
    min_delta=1e-3,
    sampler="mixed",
    bio_fraction=0.5,
    num_workers=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Set training options at the top of this file, not as arguments."
        )

    print("Loading and featurizing the shared split...", flush=True)
    df = load_benchmark_split()
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
        checkpoint_dir=AI_DIR / "checkpoints",
        history_dir=AI_DIR / "outputs" / model_name,
        verbose=True,
    )
    summary = run_mufu_benchmark(
        fit_predict,
        model_name=model_name,
        encoder=ENCODER,
        params=benchmark_params(config, smiles_encoder),
        seeds=seeds,
        df=df,
    )
    columns = ["dataset", "endpoint", "split", "metric", "value", "std", "n_seeds"]
    print(summary[columns].round(4).to_string(index=False))
