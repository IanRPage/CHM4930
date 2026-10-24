"""
Training and prediction loop for MuFu (`fit` and `predict`).

Early stopping tracks the unweighted sum of split-level validation task losses
rather than the uncertainty-weighted objective, since learned `log_vars` drift
across epochs and would rescale the weighted total independent of error. It also
drops `pos_weight`, which would let confident misses on rare positives dominate.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mufu.contract import COLUMN_GROUPS, MuFuBatch, MuFuOutputs
from mufu.data.dataset import MuFuDataset, collate, to_device
from mufu.data.sampler import AlternatingBatchSampler, MixedBatchSampler
from mufu.loss.task_loss import TaskLoss
from mufu.loss.uncertainty_weighting import UncertaintyWeighting
from mufu.model import MuFu


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    min_delta: float = 0.0
    sampler: str = "mixed"
    bio_fraction: float = 0.5
    num_workers: int = 0
    device: torch.device | str | None = None


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    weighting: UncertaintyWeighting


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


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
) -> torch.Tensor | None:
    optimizer.zero_grad(set_to_none=True)
    task_losses = task_loss(model(batch), batch)
    present = [loss for loss in task_losses.values() if loss is not None]
    if not present:
        return None
    weighting(task_losses).backward()
    optimizer.step()
    return sum(present).detach()


@torch.no_grad()
def _forward_dataset(model: MuFu, dataset: MuFuDataset, batch_size: int) -> MuFuOutputs:
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    batches = [model(to_device(batch, device, non_blocking=True)) for batch in loader]
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
    *,
    verbose: bool = False,
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
        datasets["train"],
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.to(device)
    task_loss = TaskLoss.from_dataset(datasets["train"]).to(device)
    val_task_loss = TaskLoss().to(device)
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
                model,
                to_device(batch, device, non_blocking=True),
                task_loss,
                weighting,
                optimizer,
            )
            if loss is not None:
                step_losses.append(loss)

        train_losses.append(
            (sum(step_losses) / len(step_losses)).item()
            if step_losses
            else float("nan")
        )
        val_loss = evaluate_val_loss(
            model, datasets["val"], val_task_loss, cfg.batch_size
        )
        val_losses.append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train={train_losses[-1]:.4f} | val={val_loss:.4f}",
                flush=True,
            )

        if best_state is None or val_loss < best_val_loss - cfg.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {
                "model": cpu_state(model),
                "weighting": cpu_state(weighting),
            }
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
) -> dict[str, list[str] | np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("predict needs a non-empty dataset")
    outputs = _forward_dataset(model, dataset, batch_size)
    for key in ("active_logits", "tox_logits"):
        outputs[key] = torch.sigmoid(outputs[key])
    return {
        "smiles": list(dataset.smiles),
        **{
            col: outputs[key][:, j].numpy().astype(float, copy=False)
            for key, columns in COLUMN_GROUPS
            for j, col in enumerate(columns)
        },
    }
