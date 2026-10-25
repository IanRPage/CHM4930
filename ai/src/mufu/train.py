"""
Training and prediction loop for MuFu (`fit` and `predict`).

Early stopping ignores `pos_weight` and uncertainty weighting, so rare-positive misses
and drifting `log_vars` can't dominate it. `train_loss` is logged the same way, but
with dropout on, so it won't match `val_loss` exactly. In `task_family` mode bio and
tox each keep a best state and patience; `fit` ends on the bio state and returns the
tox state in `family_states`, which `predict` uses for `tox_logits`.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation import evaluation_metrics
from mufu.calibration import CALIBRATION_MODES, CalibrationParams
from mufu.contract import (
    COLUMN_GROUPS,
    N_BIO,
    TASKS,
    TOX_COLUMNS,
    MuFuBatch,
    MuFuOutputs,
)
from mufu.data.dataset import MuFuDataset, collate, to_device
from mufu.data.sampler import AlternatingBatchSampler, MixedBatchSampler
from mufu.loss.task_loss import POS_WEIGHT_MODES, TaskLoss, compute_task_losses
from mufu.loss.uncertainty_weighting import UncertaintyWeighting
from mufu.model import MuFu
from pipeline.toxicity_loader import DATASETS

CONFIG_OPTIONS = {
    "sampler": ("mixed", "alternating"),
    "checkpoint_mode": ("single", "task_family"),
    "pos_weight": POS_WEIGHT_MODES,
    "stopping": ("val_loss", "norm_loss", "metric"),
    "calibration": CALIBRATION_MODES,
}
FAMILY_TASKS = {"bio": ("pic50", "bio_cls"), "tox": ("tox",)}
TOX21_INDICES = [TOX_COLUMNS.index(t) for t in DATASETS["tox21"]["tasks"]]


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
    checkpoint_mode: str = "single"
    pos_weight: str = "raw"
    stopping: str = "val_loss"
    log_var_lr: float | None = None
    task_weights: tuple[float, float, float] | None = None
    fp_hidden_dims: tuple[int, ...] = (1024, 512)
    fp_dropout: float = 0.2
    gnn_hidden_dims: tuple[int, ...] = (256, 256, 128)
    gnn_dropout: float = 0.1
    calibration: str = "none"
    num_workers: int = 0
    device: torch.device | str | None = None


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    weighting: UncertaintyWeighting
    val_task_losses: dict[str, list[float]] = field(
        default_factory=lambda: {t: [] for t in TASKS}
    )
    family_states: dict[str, dict[str, torch.Tensor]] | None = None
    best_epochs: dict[str, int] | None = None


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
        if not k.startswith("smiles_encoder.")
    }


def load_non_smiles_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    incompatible = module.load_state_dict(state, strict=False)
    bad_missing = [
        k for k in incompatible.missing_keys if not k.startswith("smiles_encoder.")
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"invalid state_dict: missing={bad_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


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
                "lr": cfg.lr if cfg.log_var_lr is None else cfg.log_var_lr,
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
    *,
    task_weights: tuple[float, float, float] | None = None,
) -> torch.Tensor | None:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch)
    task_losses = task_loss(outputs, batch)
    if all(loss is None for loss in task_losses.values()):
        return None
    if task_weights is None:
        objective = weighting(task_losses)
    else:
        objective = torch.stack(
            [
                float(w) * task_losses[t]
                for w, t in zip(task_weights, TASKS)
                if task_losses[t] is not None
            ]
        ).sum()
    objective.backward()
    optimizer.step()

    with torch.no_grad():
        unweighted = compute_task_losses(outputs, batch)
    return sum(loss for loss in unweighted.values() if loss is not None)


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
) -> tuple[MuFuOutputs, dict[str, float]]:
    outputs = _forward_dataset(model, val_dataset, batch_size)
    losses = task_loss(outputs, {**val_dataset.labels, **val_dataset.masks})
    return outputs, {
        t: float("nan") if loss is None else loss.item() for t, loss in losses.items()
    }


def _sum_losses(
    losses: Mapping[str, float],
    tasks: tuple[str, ...],
    scales: Mapping[str, float] | None = None,
) -> float:
    terms = [
        losses[t] / (1.0 if scales is None else scales[t])
        for t in tasks
        if np.isfinite(losses[t])
    ]
    return float(sum(terms)) if terms else float("inf")


def _metric_signals(outputs: MuFuOutputs, val_dataset: MuFuDataset) -> dict[str, float]:
    y = val_dataset.labels["y_pic50"].cpu().numpy()
    mask = val_dataset.masks["mask_bio"].cpu().numpy()
    pred = outputs["pic50"].numpy()
    rmses = [
        evaluation_metrics(
            y[mask[:, j], j], pred[mask[:, j], j], task_type="regression"
        )["rmse"]
        for j in range(N_BIO)
        if mask[:, j].any()
    ]
    rmses = [r for r in rmses if np.isfinite(r)]

    auc = evaluation_metrics(
        val_dataset.labels["y_tox"].cpu().numpy()[:, TOX21_INDICES],
        torch.sigmoid(outputs["tox_logits"][:, TOX21_INDICES]).numpy(),
        task_type="multitask",
        mask=val_dataset.masks["mask_tox"].cpu().numpy()[:, TOX21_INDICES],
    )["mean_roc_auc"]
    return {
        "bio": float(np.mean(rmses)) if rmses else float("inf"),
        "tox": -float(auc) if np.isfinite(auc) else float("inf"),
    }


def _validate_config(cfg: TrainConfig) -> None:
    for name, allowed in CONFIG_OPTIONS.items():
        value = getattr(cfg, name)
        if value not in allowed:
            raise ValueError(f"unknown {name} {value!r}; expected one of {allowed}")
    if cfg.stopping == "metric" and cfg.checkpoint_mode != "task_family":
        raise ValueError("stopping='metric' requires checkpoint_mode='task_family'")
    if cfg.task_weights is not None and len(cfg.task_weights) != len(TASKS):
        raise ValueError(
            f"task_weights: expected {len(TASKS)} weights for {TASKS}, "
            f"got {len(cfg.task_weights)}"
        )


def fit(
    model: MuFu,
    datasets: Mapping[str, MuFuDataset],
    config: TrainConfig | None = None,
    *,
    verbose: bool = False,
) -> TrainResult:
    cfg = config or TrainConfig()
    _validate_config(cfg)
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if cfg.sampler == "mixed":
        sampler = MixedBatchSampler(datasets["train"], cfg.batch_size, cfg.bio_fraction)
    else:
        sampler = AlternatingBatchSampler(datasets["train"], cfg.batch_size)
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.to(device)
    task_loss = TaskLoss.from_dataset(datasets["train"], pos_weight=cfg.pos_weight).to(
        device
    )
    val_task_loss = TaskLoss().to(device)
    weighting = UncertaintyWeighting().to(device)
    optimizer = make_optimizer(model, weighting, cfg)

    families = FAMILY_TASKS if cfg.checkpoint_mode == "task_family" else {"all": TASKS}
    primary = next(iter(families))
    best_signals = dict.fromkeys(families, float("inf"))
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    best_epochs = dict.fromkeys(families, 0)
    stalled = dict.fromkeys(families, 0)
    best_val_loss = float("inf")
    best_weighting: dict[str, torch.Tensor] = {}
    epoch1_losses: dict[str, float] = {}
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_task_losses: dict[str, list[float]] = {t: [] for t in TASKS}

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
                task_weights=cfg.task_weights,
            )
            if loss is not None:
                step_losses.append(loss)
        train_losses.append(
            (sum(step_losses) / len(step_losses)).item()
            if step_losses
            else float("nan")
        )

        outputs, losses = evaluate_val_loss(
            model, datasets["val"], val_task_loss, cfg.batch_size
        )
        val_losses.append(_sum_losses(losses, TASKS))
        for t in TASKS:
            val_task_losses[t].append(losses[t])
        if not epoch1_losses:
            epoch1_losses = {
                t: v if np.isfinite(v) and v > 0.0 else 1.0 for t, v in losses.items()
            }

        if verbose:
            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train={train_losses[-1]:.4f} | val={val_losses[-1]:.4f}",
                flush=True,
            )

        if cfg.stopping == "metric":
            signals = _metric_signals(outputs, datasets["val"])
        else:
            scales = epoch1_losses if cfg.stopping == "norm_loss" else None
            signals = {f: _sum_losses(losses, ts, scales) for f, ts in families.items()}

        snapshot = None
        for fam in families:
            if fam in best_states and signals[fam] >= best_signals[fam] - cfg.min_delta:
                stalled[fam] += 1
                continue
            snapshot = snapshot or cpu_state(model)
            best_signals[fam] = signals[fam]
            best_states[fam] = snapshot
            best_epochs[fam] = epoch
            stalled[fam] = 0
            if fam == primary:
                best_val_loss = _sum_losses(losses, families[primary])
                best_weighting = cpu_state(weighting)
        if all(n >= cfg.patience for n in stalled.values()):
            break

    if best_states:
        load_non_smiles_state(model, best_states[primary])
        weighting.load_state_dict(best_weighting)
    family_states = {f: s for f, s in best_states.items() if f != primary}
    return TrainResult(
        best_epoch=best_epochs[primary],
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        weighting=weighting,
        val_task_losses=val_task_losses,
        family_states=family_states or None,
        best_epochs=best_epochs if cfg.checkpoint_mode == "task_family" else None,
    )


def predict_logits(
    model: MuFu,
    dataset: MuFuDataset,
    *,
    batch_size: int = 64,
    family_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> MuFuOutputs:
    if len(dataset) == 0:
        raise ValueError("predict needs a non-empty dataset")
    outputs = _forward_dataset(model, dataset, batch_size)
    if family_states:
        tox_model = copy.deepcopy(
            model, memo={id(model.smiles_encoder): model.smiles_encoder}
        )
        load_non_smiles_state(tox_model, family_states["tox"])
        outputs["tox_logits"] = _forward_dataset(tox_model, dataset, batch_size)[
            "tox_logits"
        ]
    return outputs


def predict(
    model: MuFu,
    dataset: MuFuDataset,
    *,
    batch_size: int = 64,
    family_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    calibration: CalibrationParams | None = None,
) -> dict[str, list[str] | np.ndarray]:
    outputs = predict_logits(
        model, dataset, batch_size=batch_size, family_states=family_states
    )
    if calibration is not None:
        outputs = calibration.apply(outputs)
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
