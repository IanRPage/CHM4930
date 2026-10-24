"""
Run MuFu through the shared benchmark (`baseline.benchmark.run_benchmark`).

`make_fit_predict` builds the per-seed callable from prepared datasets and one shared
SMILES encoder, so a frozen encoder's embedding cache carries across seeds.
`DEFAULT_CONFIG` and `ENCODER` are the settings the script and notebook both start from.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from baseline.results import file_sha256, git_commit
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import D_MODEL, MODALITY_DROPOUT_P, Encoder
from mufu.data.dataset import MuFuDataset
from mufu.model import MuFu
from mufu.train import TrainConfig, cpu_state, fit, predict
from pipeline.bioactivity_loader import PIC50_ACTIVE_THRESHOLD
from pipeline.combined_loader import COMBINED_CSV_PATH

ENCODER = "molformer-ecfp4-gine"
DEFAULT_CONFIG = TrainConfig(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=50,
    patience=10,
    min_delta=1e-3,
    sampler="mixed",
    bio_fraction=0.5,
)
RUNTIME_FIELDS = ("device", "num_workers")


def benchmark_params(cfg: TrainConfig, smiles_encoder: Encoder) -> dict:
    name = getattr(smiles_encoder, "model_name", type(smiles_encoder).__name__)
    revision = getattr(smiles_encoder, "revision", None)
    train = {k: v for k, v in asdict(cfg).items() if k not in RUNTIME_FIELDS}
    return {
        **train,
        "d_model": D_MODEL,
        "modality_dropout_p": MODALITY_DROPOUT_P,
        "active_threshold": PIC50_ACTIVE_THRESHOLD,
        "smiles_encoder": f"{name}@{revision}" if revision else name,
        "smiles_freeze": getattr(smiles_encoder, "freeze", None),
    }


def make_fit_predict(
    datasets: Mapping[str, MuFuDataset],
    smiles_encoder: Encoder,
    cfg: TrainConfig,
    *,
    model_name: str,
    checkpoint_dir: Path,
    history_dir: Path,
    data_path: Path = COMBINED_CSV_PATH,
    verbose: bool = False,
) -> Callable[[pd.DataFrame, int], dict]:
    params = benchmark_params(cfg, smiles_encoder)

    # df unused. datasets built from it once, run_benchmark checks returned
    # SMILES against each split
    def fit_predict(df: pd.DataFrame, seed: int) -> dict:
        model = MuFu(smiles_encoder, FingerprintEncoder(), GNNEncoder())
        result = fit(model, datasets, cfg, verbose=verbose)
        if verbose:
            print(
                f"seed {seed}: best epoch {result.best_epoch} | "
                f"val loss {result.best_val_loss:.4f}",
                flush=True,
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": cpu_state(model),
                "weighting": cpu_state(result.weighting),
                "epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "seed": int(seed),
                "params": params,
                "git_commit": git_commit(),
                "data_sha256": file_sha256(data_path),
            },
            checkpoint_dir / f"{model_name}_seed_{seed}.pt",
        )

        history_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": range(1, len(result.train_losses) + 1),
                "train_loss": result.train_losses,
                "val_loss": result.val_losses,
            }
        ).to_csv(history_dir / f"seed_{seed}_loss.csv", index=False)

        return {
            split: predict(model, datasets[split], batch_size=cfg.batch_size)
            for split in ("val", "test")
        }

    return fit_predict
