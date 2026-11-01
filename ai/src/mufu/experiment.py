"""
Run MuFu through the shared benchmark.

`make_fit_predict` builds the `fit_predict` that `run_benchmark` calls per seed, reusing
one SMILES encoder so its embedding cache carries across seeds. `DEFAULT_CONFIG` and
`ENCODER` are the script and notebook defaults.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from baseline.results import file_sha256, git_commit
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.calibration import CalibrationParams, fit_calibration
from mufu.contract import D_MODEL, MODALITY_DROPOUT_P, Encoder
from mufu.data.dataset import MuFuDataset
from mufu.loss.task_loss import dataset_pos_weights
from mufu.mc_dropout import (
    empirical_coverage,
    fit_mc_dropout,
    predict_with_intervals,
)
from mufu.model import MuFu
from mufu.train import (
    TrainConfig,
    cpu_state,
    fit,
    load_non_smiles_state,
    predict,
    predict_logits,
)
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
    bio_fraction=0.8,
    checkpoint_mode="task_family",
    pos_weight="raw",
    stopping="metric",
    log_var_lr=None,
    task_weights=(1.0, 0.5, 0.5),
    fp_hidden_dims=(512, 256),
    fp_dropout=0.6,
    calibration="platt",
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


# checkpoints from before gnn_hidden_dims was recorded in params
def infer_gnn_hidden_dims(state: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
    dims = []
    while (key := f"graph_encoder.mp_layers.{len(dims)}.nn.0.weight") in state:
        dims.append(state[key].shape[0])
    if not dims:
        raise ValueError("checkpoint has no graph_encoder.mp_layers weights")
    return tuple(dims)


def load_mufu_checkpoint(
    path: Path,
    smiles_encoder: Encoder,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MuFu, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    params = ckpt.get("params", {})
    fp_hidden_dims = tuple(params.get("fp_hidden_dims", (1024, 512)))
    fp_dropout = float(params.get("fp_dropout", 0.2))
    gnn_hidden_dims = tuple(
        params.get("gnn_hidden_dims") or infer_gnn_hidden_dims(ckpt["model"])
    )
    gnn_dropout = float(params.get("gnn_dropout", 0.1))

    model = MuFu(
        smiles_encoder,
        FingerprintEncoder(hidden_dims=fp_hidden_dims, dropout=fp_dropout),
        GNNEncoder(hidden_dims=gnn_hidden_dims, dropout=gnn_dropout),
    ).to(device)
    load_non_smiles_state(model, ckpt["model"])
    model.eval()

    cal_state = ckpt.get("calibration")
    calibration = cal_state and CalibrationParams.from_state_dict(cal_state)
    return model, {
        "family_states": ckpt.get("family_states"),
        "best_epochs": ckpt.get("best_epochs"),
        "calibration": calibration,
        "checkpoint": ckpt,
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
    if any(p.requires_grad for p in smiles_encoder.parameters()):
        raise ValueError("smiles_encoder must be frozen; it's shared across seeds")
    params = benchmark_params(cfg, smiles_encoder)

    # df unused, datasets already built from it and run_benchmark checks smiles
    def fit_predict(df: pd.DataFrame, seed: int) -> dict:
        model = MuFu(
            smiles_encoder,
            FingerprintEncoder(hidden_dims=cfg.fp_hidden_dims, dropout=cfg.fp_dropout),
            GNNEncoder(hidden_dims=cfg.gnn_hidden_dims, dropout=cfg.gnn_dropout),
        )
        result = fit(model, datasets, cfg, verbose=verbose)
        if verbose:
            print(
                f"seed {seed}: best epoch {result.best_epochs or result.best_epoch} | "
                f"val loss {result.best_val_loss:.4f}",
                flush=True,
            )

        calibration = None
        if cfg.calibration != "none":
            calibration = fit_calibration(
                cfg.calibration,
                predict_logits(
                    model,
                    datasets["val"],
                    batch_size=cfg.batch_size,
                    family_states=result.family_states,
                ),
                datasets["val"],
                pos_weights=dataset_pos_weights(datasets["train"], mode=cfg.pos_weight),
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{model_name}_seed_{seed}.pt"
        torch.save(
            {
                "model": cpu_state(model),
                "family_states": result.family_states,
                "best_epochs": result.best_epochs,
                "calibration": calibration and calibration.state_dict(),
                "weighting": cpu_state(result.weighting),
                "epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "seed": int(seed),
                "params": params,
                "git_commit": git_commit(),
                "data_sha256": file_sha256(data_path),
            },
            ckpt_path,
        )

        conformal = fit_mc_dropout(
            model, datasets["val"], ckpt_path, batch_size=cfg.batch_size
        )

        history_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": range(1, len(result.train_losses) + 1),
                "train_loss": result.train_losses,
                "val_loss": result.val_losses,
                "val_pic50_loss": result.val_task_losses["pic50"],
                "val_bio_cls_loss": result.val_task_losses["bio_cls"],
                "val_tox_loss": result.val_task_losses["tox"],
            }
        ).to_csv(history_dir / f"seed_{seed}_loss.csv", index=False)
        empirical_coverage(
            predict_with_intervals(
                model, datasets["test"], conformal, batch_size=cfg.batch_size
            ),
            datasets["test"],
            conformal.coverages,
        ).to_csv(history_dir / f"seed_{seed}_mc_dropout_coverage.csv", index=False)

        return {
            split: predict(
                model,
                datasets[split],
                batch_size=cfg.batch_size,
                family_states=result.family_states,
                calibration=calibration,
            )
            for split in ("val", "test")
        }

    return fit_predict
