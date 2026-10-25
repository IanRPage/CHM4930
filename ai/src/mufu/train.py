"""
Training and prediction loop for MuFu (`fit` and `predict`).

Early stopping tracks validation performance without `pos_weight` (which would let
confident misses on rare positives dominate) and without uncertainty weighting (since
learned `log_vars` drift across epochs). Note that logged `train_loss` is measured
with encoder dropout and modality dropout active, so it remains slightly offset from
`val_loss` even without `pos_weight`.
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

CHECKPOINT_MODES = ("single", "task_family")
STOPPING_MODES = ("val_loss", "norm_loss", "metric")
FAMILY_TASKS: dict[str, tuple[str, ...]] = {
    "bio": ("pic50", "bio_cls"),
    "tox": ("tox",),
}
FAMILY_OUTPUT_KEYS: dict[str, tuple[str, ...]] = {
    "bio": ("pic50", "active_logits"),
    "tox": ("tox_logits",),
}
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
    log_var_lr = cfg.lr if cfg.log_var_lr is None else cfg.log_var_lr
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": list(weighting.log_vars.values()),
                "weight_decay": 0.0,
                "lr": log_var_lr,
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
    present = [loss for loss in task_losses.values() if loss is not None]
    if not present:
        return None
    if task_weights is None:
        objective = weighting(task_losses)
    else:
        if len(task_weights) != len(TASKS):
            raise ValueError(
                f"task_weights: expected {len(TASKS)} weights for {TASKS}, "
                f"got {len(task_weights)}"
            )
        terms = [
            float(w) * task_losses[t]
            for w, t in zip(task_weights, TASKS)
            if task_losses[t] is not None
        ]
        objective = torch.stack(terms).sum()
    objective.backward()
    optimizer.step()

    detached = {k: v.detach() for k, v in outputs.items()}
    unweighted = compute_task_losses(detached, batch)
    unweighted_present = [loss for loss in unweighted.values() if loss is not None]
    return sum(unweighted_present).detach()


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
    losses = task_loss(outputs, val_batch)
    task_loss._last_outputs = outputs
    task_loss._last_losses = losses
    present = [loss for loss in losses.values() if loss is not None]
    return sum(present).item() if present else float("inf")


def _compute_val_metric_signals(
    outputs: MuFuOutputs, val_dataset: MuFuDataset
) -> dict[str, float]:
    y_pic50 = val_dataset.labels["y_pic50"].detach().cpu().numpy()
    mask_bio = val_dataset.masks["mask_bio"].detach().cpu().numpy()
    pred_pic50 = outputs["pic50"].detach().cpu().numpy()

    rmses: list[float] = []
    for j in range(N_BIO):
        valid = mask_bio[:, j]
        if valid.any():
            reg = evaluation_metrics(
                y_pic50[valid, j], pred_pic50[valid, j], task_type="regression"
            )
            if np.isfinite(reg["rmse"]):
                rmses.append(float(reg["rmse"]))
    bio_rmse = float(np.mean(rmses)) if rmses else float("inf")

    y_tox21 = val_dataset.labels["y_tox"].detach().cpu().numpy()[:, TOX21_INDICES]
    mask_tox21 = val_dataset.masks["mask_tox"].detach().cpu().numpy()[:, TOX21_INDICES]
    tox21_probs = (
        torch.sigmoid(outputs["tox_logits"][:, TOX21_INDICES]).detach().cpu().numpy()
    )
    tox21_res = evaluation_metrics(
        y_tox21, tox21_probs, task_type="multitask", mask=mask_tox21
    )
    mean_auc = tox21_res["mean_roc_auc"]
    tox_signal = -float(mean_auc) if np.isfinite(mean_auc) else float("inf")

    return {"bio": bio_rmse, "tox": tox_signal}


def _validate_config(cfg: TrainConfig) -> None:
    if cfg.sampler not in ("mixed", "alternating"):
        raise ValueError(
            f"unknown sampler {cfg.sampler!r}; expected 'mixed' or 'alternating'"
        )
    if cfg.checkpoint_mode not in CHECKPOINT_MODES:
        raise ValueError(
            f"unknown checkpoint_mode {cfg.checkpoint_mode!r}; "
            f"expected one of {CHECKPOINT_MODES}"
        )
    if cfg.pos_weight not in POS_WEIGHT_MODES:
        raise ValueError(
            f"unknown pos_weight {cfg.pos_weight!r}; expected one of {POS_WEIGHT_MODES}"
        )
    if cfg.stopping not in STOPPING_MODES:
        raise ValueError(
            f"unknown stopping {cfg.stopping!r}; expected one of {STOPPING_MODES}"
        )
    if cfg.stopping == "metric" and cfg.checkpoint_mode != "task_family":
        raise ValueError("stopping='metric' requires checkpoint_mode='task_family'")
    if cfg.calibration not in CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration {cfg.calibration!r}; "
            f"expected one of {CALIBRATION_MODES}"
        )
    if cfg.task_weights is not None and len(cfg.task_weights) != len(TASKS):
        raise ValueError(
            f"task_weights: expected {len(TASKS)} weights for {TASKS}, "
            f"got {len(cfg.task_weights)}"
        )


def _sum_task_subset(
    task_losses: Mapping[str, float],
    tasks: tuple[str, ...],
    scales: Mapping[str, float] | None = None,
) -> float:
    terms = []
    for t in tasks:
        val = task_losses.get(t, float("nan"))
        if np.isfinite(val):
            scale = 1.0 if scales is None else scales.get(t, 1.0)
            terms.append(val / scale)
    return float(sum(terms)) if terms else float("inf")


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

    best_val_loss = float("inf")
    best_stop_signal = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_without_improve = 0

    best_family_signals = {"bio": float("inf"), "tox": float("inf")}
    best_family_states: dict[str, dict[str, torch.Tensor] | None] = {
        "bio": None,
        "tox": None,
    }
    best_epochs = {"bio": 0, "tox": 0}
    best_weighting_state: dict[str, torch.Tensor] | None = None
    stalled = {"bio": 0, "tox": 0}
    epoch1_scales: dict[str, float] | None = None

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
        val_task_loss._last_losses = None
        val_task_loss._last_outputs = None
        val_loss = evaluate_val_loss(
            model, datasets["val"], val_task_loss, cfg.batch_size
        )
        val_losses.append(val_loss)

        raw_last_losses = getattr(val_task_loss, "_last_losses", None)
        if raw_last_losses is not None:
            step_val_losses = {
                t: (
                    float(raw_last_losses[t].item())
                    if raw_last_losses.get(t) is not None
                    else float("nan")
                )
                for t in TASKS
            }
        else:
            step_val_losses = {
                "pic50": val_loss,
                "bio_cls": float("nan"),
                "tox": float("nan"),
            }
        for t in TASKS:
            val_task_losses[t].append(step_val_losses[t])

        if epoch1_scales is None:
            epoch1_scales = {
                t: (
                    step_val_losses[t]
                    if np.isfinite(step_val_losses[t]) and step_val_losses[t] > 0.0
                    else 1.0
                )
                for t in TASKS
            }

        if verbose:
            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train={train_losses[-1]:.4f} | val={val_loss:.4f}",
                flush=True,
            )

        if cfg.checkpoint_mode == "single":
            if cfg.stopping == "val_loss":
                stop_signal = val_loss
            else:
                stop_signal = _sum_task_subset(
                    step_val_losses, TASKS, scales=epoch1_scales
                )

            if best_state is None or stop_signal < best_stop_signal - cfg.min_delta:
                best_stop_signal = stop_signal
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
        else:
            bio_unweighted = _sum_task_subset(
                step_val_losses, FAMILY_TASKS["bio"], scales=None
            )
            if cfg.stopping == "val_loss":
                family_signals = {
                    fam: _sum_task_subset(step_val_losses, tasks, scales=None)
                    for fam, tasks in FAMILY_TASKS.items()
                }
            elif cfg.stopping == "norm_loss":
                family_signals = {
                    fam: _sum_task_subset(step_val_losses, tasks, scales=epoch1_scales)
                    for fam, tasks in FAMILY_TASKS.items()
                }
            else:
                last_outputs = getattr(val_task_loss, "_last_outputs", None)
                if last_outputs is None:
                    last_outputs = _forward_dataset(
                        model, datasets["val"], cfg.batch_size
                    )
                family_signals = _compute_val_metric_signals(
                    last_outputs, datasets["val"]
                )

            snapshot: dict[str, torch.Tensor] | None = None
            for fam in ("bio", "tox"):
                sig = family_signals[fam]
                if (
                    best_family_states[fam] is None
                    or sig < best_family_signals[fam] - cfg.min_delta
                ):
                    if snapshot is None:
                        snapshot = cpu_state(model)
                    best_family_signals[fam] = sig
                    best_family_states[fam] = snapshot
                    best_epochs[fam] = epoch
                    stalled[fam] = 0
                    if fam == "bio":
                        best_val_loss = bio_unweighted
                        best_epoch = epoch
                        best_weighting_state = cpu_state(weighting)
                else:
                    stalled[fam] += 1

            if all(count >= cfg.patience for count in stalled.values()):
                break

    if cfg.checkpoint_mode == "single":
        if best_state is not None:
            load_non_smiles_state(model, best_state["model"])
            weighting.load_state_dict(best_state["weighting"])
        return TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            weighting=weighting,
            val_task_losses=val_task_losses,
            family_states=None,
            best_epochs=None,
        )

    if best_family_states["bio"] is not None:
        load_non_smiles_state(model, best_family_states["bio"])
    if best_weighting_state is not None:
        weighting.load_state_dict(best_weighting_state)

    family_states = (
        {"tox": best_family_states["tox"]}
        if best_family_states["tox"] is not None
        else None
    )
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        weighting=weighting,
        val_task_losses=val_task_losses,
        family_states=family_states,
        best_epochs=dict(best_epochs),
    )


def _clone_model_with_state(model: MuFu, state: Mapping[str, torch.Tensor]) -> MuFu:
    cloned = copy.deepcopy(model, memo={id(model.smiles_encoder): model.smiles_encoder})
    load_non_smiles_state(cloned, state)
    return cloned


def predict_logits(
    model: MuFu,
    dataset: MuFuDataset,
    *,
    batch_size: int = 64,
    family_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> MuFuOutputs:
    if len(dataset) == 0:
        raise ValueError("predict needs a non-empty dataset")
    unknown = set(family_states or ()) - set(FAMILY_OUTPUT_KEYS)
    if unknown:
        raise ValueError(
            f"unknown family_states key(s) {sorted(unknown)}; "
            f"expected subset of {sorted(FAMILY_OUTPUT_KEYS)}"
        )
    if not family_states:
        return _forward_dataset(model, dataset, batch_size)

    bio_model = (
        _clone_model_with_state(model, family_states["bio"])
        if "bio" in family_states
        else model
    )
    outputs = _forward_dataset(bio_model, dataset, batch_size)
    if "tox" in family_states:
        tox_model = _clone_model_with_state(model, family_states["tox"])
        tox_outputs = _forward_dataset(tox_model, dataset, batch_size)
        for key in FAMILY_OUTPUT_KEYS["tox"]:
            outputs[key] = tox_outputs[key]
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
