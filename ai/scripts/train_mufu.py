"""Train MuFu with the existing trainer and shared benchmark.

From ai/: python scripts/train_mufu.py
Edit the training settings below; the default runs all five shared seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from baseline.benchmark import SEEDS, load_benchmark_split, run_benchmark
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.model import MuFu
from mufu.train import TrainConfig, fit, predict

AI_DIR = Path(__file__).resolve().parents[1]

# Training settings
model_name = "mufu"
seeds = SEEDS  # Shared baseline seeds: 0, 1, 2, 3, 4.
num_epochs = 50
batch_size = 64
learn_rate = 0.001
weight_decay = 0.0001
patience = 10
sampler = "mixed"
bio_fraction = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"


def fit_predict(df, seed):
    # run_benchmark sets the random seed before calling this function.
    print(f"\nSeed {seed} | device: {device}", flush=True)

    # Model: reuse the existing encoder defaults.
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))
    fingerprint_encoder = FingerprintEncoder()
    graph_encoder = GNNEncoder()
    model = MuFu(smiles_encoder, fingerprint_encoder, graph_encoder)

    # Training: fit already handles batches, losses, early stopping, and saving.
    checkpoint = AI_DIR / "checkpoints" / f"{model_name}_seed_{seed}.pt"
    config = TrainConfig(
        lr=learn_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=num_epochs,
        patience=patience,
        sampler=sampler,
        bio_fraction=bio_fraction,
        num_workers=0,
        device=device,
        checkpoint_path=checkpoint,
    )
    result = fit(model, datasets, config, verbose=True)
    print(f"Best epoch: {result.best_epoch} | val loss: {result.best_val_loss:.4f}")
    print("Checkpoint:", checkpoint)

    # Keep only the losses needed to inspect training and draw the curves.
    history = pd.DataFrame(
        {
            "epoch": range(1, len(result.train_losses) + 1),
            "train_loss": result.train_losses,
            "val_loss": result.val_losses,
        }
    )
    output_dir = AI_DIR / "outputs" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / f"seed_{seed}_loss.csv", index=False)

    # fit has restored the best validation-selected model.
    return {
        "val": predict(model, datasets["val"], batch_size=batch_size),
        "test": predict(model, datasets["test"], batch_size=batch_size),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Set training options at the top of this file, not as arguments."
        )

    # Prepare the datasets once, not once per seed.
    print("Loading and featurizing the shared split...", flush=True)
    df = load_benchmark_split()
    datasets = split_datasets(df)
    for split in datasets:
        print(f"{split}: {len(datasets[split])} molecules")
    print(
        f"Learning rate: {learn_rate} | batch size: {batch_size} | epochs: {num_epochs}"
    )

    # The existing benchmark records metrics and this small training config.
    params = {
        "lr": learn_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "patience": patience,
        "sampler": sampler,
        "bio_fraction": bio_fraction,
        "num_workers": 0,
        "device": device,
    }
    summary = run_benchmark(
        fit_predict,
        model=model_name,
        encoder="molformer-ecfp4-gine",
        params=params,
        seeds=seeds,
        df=df,
    )
    columns = ["dataset", "endpoint", "split", "metric", "value", "std", "n_seeds"]
    print(summary[columns].round(4).to_string(index=False))
