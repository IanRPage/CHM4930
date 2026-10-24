"""
Run MuFu through the shared benchmark (`baseline.benchmark.run_benchmark`).

`make_fit_predict` builds the per-seed callable from prepared datasets and one shared
SMILES encoder, so a frozen encoder's embedding cache carries across seeds.
`run_mufu_benchmark` records each seed as soon as it finishes, so a crash keeps the
seeds already done.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch

from baseline.benchmark import run_benchmark
from baseline.results import file_sha256, git_commit
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import D_MODEL, MODALITY_DROPOUT_P, Encoder
from mufu.data.dataset import MuFuDataset
from mufu.model import MuFu
from mufu.train import TrainConfig, _cpu_state, fit, predict
from pipeline.bioactivity_loader import PIC50_ACTIVE_THRESHOLD
from pipeline.combined_loader import COMBINED_CSV_PATH

RUNTIME_FIELDS = ("device", "num_workers", "checkpoint_path")


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
    fit_cfg = replace(cfg, checkpoint_path=None)

    # df is unused: datasets are built from it once, and run_benchmark checks the
    # returned SMILES against each split.
    def fit_predict(df: pd.DataFrame, seed: int) -> dict:
        model = MuFu(smiles_encoder, FingerprintEncoder(), GNNEncoder())
        result = fit(model, datasets, fit_cfg, verbose=verbose)
        if verbose:
            print(
                f"seed {seed}: best epoch {result.best_epoch} | "
                f"val loss {result.best_val_loss:.4f}",
                flush=True,
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": _cpu_state(model),
                "weighting": _cpu_state(result.weighting),
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


def run_mufu_benchmark(
    fit_predict: Callable[[pd.DataFrame, int], dict],
    *,
    model_name: str,
    encoder: str,
    params: dict,
    seeds: Sequence[int],
    df: pd.DataFrame,
    **benchmark_kwargs,
) -> pd.DataFrame:
    if not seeds:
        raise ValueError("seeds must not be empty")
    for seed in seeds:
        summary = run_benchmark(
            fit_predict,
            model=model_name,
            encoder=encoder,
            params=params,
            seeds=(seed,),
            df=df,
            **benchmark_kwargs,
        )
    return summary
