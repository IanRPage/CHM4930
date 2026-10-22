"""
Training and prediction loop for MuFu (`fit` and `predict`).

Early stopping tracks the unweighted sum of split-level validation task losses
rather than the uncertainty-weighted objective, since learned `log_vars` drift
across epochs and would rescale the weighted total independent of error.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mufu.contract import (
    ACTIVE_COLUMNS,
    PIC50_COLUMNS,
    TOX_COLUMNS,
    MuFuBatch,
    MuFuOutputs,
)
from mufu.dataset import MuFuDataset, collate, to_device
from mufu.model import MuFu
from mufu.sampler import AlternatingBatchSampler, MixedBatchSampler
from mufu.task_loss import TaskLoss
from mufu.uncertainty_weighting import UncertaintyWeighting

COLUMN_GROUPS = (
    ("pic50", PIC50_COLUMNS),
    ("active_logits", ACTIVE_COLUMNS),
    ("tox_logits", TOX_COLUMNS),
)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    sampler: str = "mixed"
    bio_fraction: float = 0.5
    checkpoint_path: Path | None = None
    device: torch.device | str | None = None


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    weighting: UncertaintyWeighting


def make_optimizer(
    model: MuFu, weighting: UncertaintyWeighting, cfg: TrainConfig
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": list(weighting.log_vars.values()),
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.lr,
    )


def train_step(
    model: MuFu,
    batch: MuFuBatch,
    task_loss: TaskLoss,
    weighting: UncertaintyWeighting,
    optimizer: torch.optim.Optimizer,
) -> float | None:
    optimizer.zero_grad(set_to_none=True)
    task_losses = task_loss(model(batch), batch)
    if all(loss is None for loss in task_losses.values()):
        return None
    loss = weighting(task_losses)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _forward_dataset(model: MuFu, dataset: MuFuDataset, batch_size: int) -> MuFuOutputs:
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    batches = [model(to_device(batch, device)) for batch in loader]
    return {
        key: torch.cat([b[key].cpu() for b in batches], dim=0)
        for key, _ in COLUMN_GROUPS
    }


def evaluate_val_loss(
    model: MuFu,
    val_dataset: MuFuDataset,
    task_loss: TaskLoss,
    batch_size: int,
) -> float:
    outputs = _forward_dataset(model, val_dataset, batch_size)
    val_batch = {**val_dataset.labels, **val_dataset.masks}
    present = [
        loss for loss in task_loss(outputs, val_batch).values() if loss is not None
    ]
    return sum(present).item() if present else float("inf")


def fit(
    model: MuFu,
    datasets: Mapping[str, MuFuDataset],
    config: TrainConfig | None = None,
) -> TrainResult:
    cfg = config or TrainConfig()
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if cfg.sampler == "mixed":
        sampler = MixedBatchSampler(datasets["train"], cfg.batch_size, cfg.bio_fraction)
    elif cfg.sampler == "alternating":
        sampler = AlternatingBatchSampler(datasets["train"], cfg.batch_size)
    else:
        raise ValueError(
            f"unknown sampler {cfg.sampler!r}; expected 'mixed' or 'alternating'"
        )

    train_loader = DataLoader(
        datasets["train"], batch_sampler=sampler, collate_fn=collate
    )
    model.to(device)
    task_loss = TaskLoss.from_dataset(datasets["train"]).to(device)
    weighting = UncertaintyWeighting().to(device)
    optimizer = make_optimizer(model, weighting, cfg)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_without_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        weighting.train()
        step_losses = []
        for batch in train_loader:
            loss = train_step(
                model, to_device(batch, device), task_loss, weighting, optimizer
            )
            if loss is not None:
                step_losses.append(loss)

        train_losses.append(
            sum(step_losses) / len(step_losses) if step_losses else float("nan")
        )
        val_loss = evaluate_val_loss(model, datasets["val"], task_loss, cfg.batch_size)
        val_losses.append(val_loss)

        if best_state is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "weighting": copy.deepcopy(weighting.state_dict()),
                "epoch": epoch,
            }
            if cfg.checkpoint_path is not None:
                path = Path(cfg.checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        weighting.load_state_dict(best_state["weighting"])

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        weighting=weighting,
    )


def predict(
    model: MuFu,
    dataset: MuFuDataset,
    *,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("predict needs a non-empty dataset")
    outputs = _forward_dataset(model, dataset, batch_size)
    return {
        col: outputs[key][:, j].numpy().astype(float, copy=False)
        for key, columns in COLUMN_GROUPS
        for j, col in enumerate(columns)
    }
