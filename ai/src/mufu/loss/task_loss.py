"""
Per-task masked-mean loss functions and class-imbalance weighting for MuFu.

Each task's loss is averaged first across the columns a compound is labeled on
then across labeled compounds in the batch. Tasks with no labels in the batch
return None so uncertainty weighting skips them cleanly.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mufu.contract import (
    N_BIO,
    N_TOX,
    TASK_SPECS,
    TASKS,
    MuFuBatch,
    MuFuOutputs,
    TaskLosses,
    check_tensor,
    validate_outputs,
)
from mufu.data.dataset import MuFuDataset

POS_WEIGHT_MODES = ("raw", "cap5", "cap10", "sqrt", "none")
POS_WEIGHT_CAPS = {"cap5": 5.0, "cap10": 10.0}


# capN clamps to [1/N, N], none drops pos_weight
def transform_pos_weight(
    pos_weight: torch.Tensor, mode: str = "raw"
) -> torch.Tensor | None:
    if mode not in POS_WEIGHT_MODES:
        raise ValueError(
            f"unknown pos_weight {mode!r}; expected one of {POS_WEIGHT_MODES}"
        )
    if mode in POS_WEIGHT_CAPS:
        cap = POS_WEIGHT_CAPS[mode]
        return pos_weight.clamp(min=1.0 / cap, max=cap)
    if mode == "sqrt":
        return torch.sqrt(pos_weight)
    return pos_weight if mode == "raw" else None


# averages over valid columns per compound, then over compounds with >=1 label
def masked_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    if losses.shape != mask.shape:
        raise ValueError(
            f"losses shape {tuple(losses.shape)} != mask shape {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"mask: expected dtype bool, got {mask.dtype}")

    col_counts = mask.sum(dim=1)
    valid_rows = col_counts > 0
    if not valid_rows.any():
        return None

    masked_losses = torch.where(mask, losses, 0.0)
    per_compound = masked_losses[valid_rows].sum(dim=1) / col_counts[valid_rows]
    return per_compound.mean()


# computes per-column neg/pos ratio over masked training labels
def compute_pos_weight(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if labels.shape != mask.shape or labels.ndim != 2:
        raise ValueError(
            f"expected matching 2D (N, C) labels and mask, got "
            f"{tuple(labels.shape)} and {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"mask: expected dtype bool, got {mask.dtype}")

    pos = torch.where(mask, labels, 0.0).sum(dim=0)
    neg = torch.where(mask, 1.0 - labels, 0.0).sum(dim=0)
    both_present = (pos > 0) & (neg > 0)
    safe_pos = torch.where(both_present, pos, torch.ones_like(pos))
    return torch.where(both_present, neg / safe_pos, torch.ones_like(pos))


# computes pos_weight vectors for bio_cls and tox from a training MuFuDataset
def dataset_pos_weights(
    dataset: MuFuDataset, mode: str = "raw"
) -> dict[str, torch.Tensor | None]:
    raw = {
        "bio_cls": compute_pos_weight(
            dataset.labels["y_active"], dataset.masks["mask_bio"]
        ),
        "tox": compute_pos_weight(dataset.labels["y_tox"], dataset.masks["mask_tox"]),
    }
    return {k: transform_pos_weight(v, mode=mode) for k, v in raw.items()}


# masked MSE loss for pIC50 regression
def pic50_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor | None:
    safe_target = torch.where(mask, target, 0.0)
    losses = F.mse_loss(pred, safe_target, reduction="none")
    return masked_mean(losses, mask)


# masked weighted BCE-with-logits loss for multi-target binary classification
def bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if pos_weight is not None:
        if pos_weight.shape != (logits.shape[1],):
            raise ValueError(
                f"pos_weight: expected shape {(logits.shape[1],)}, got "
                f"{tuple(pos_weight.shape)}"
            )
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    safe_target = torch.where(mask, target, 0.0)
    losses = F.binary_cross_entropy_with_logits(
        logits, safe_target, pos_weight=pos_weight, reduction="none"
    )
    return masked_mean(losses, mask)


# computes batch-normalized masked-mean loss for each MuFu task
def compute_task_losses(
    outputs: MuFuOutputs,
    batch: MuFuBatch,
    pos_weights: dict[str, torch.Tensor] | None = None,
) -> TaskLosses:
    batch_size = outputs["pic50"].shape[0]
    validate_outputs(outputs, batch_size)
    for task in TASKS:
        spec = TASK_SPECS[task]
        width = outputs[spec.output].shape[1]
        check_tensor(spec.label, batch[spec.label], (batch_size, width), "float")
        check_tensor(spec.mask, batch[spec.mask], (batch_size, width), "bool")

    weights = pos_weights or {}
    return {
        "pic50": pic50_loss(outputs["pic50"], batch["y_pic50"], batch["mask_bio"]),
        "bio_cls": bce_loss(
            outputs["active_logits"],
            batch["y_active"],
            batch["mask_bio"],
            pos_weight=weights.get("bio_cls"),
        ),
        "tox": bce_loss(
            outputs["tox_logits"],
            batch["y_tox"],
            batch["mask_tox"],
            pos_weight=weights.get("tox"),
        ),
    }


class TaskLoss(nn.Module):
    def __init__(
        self,
        bio_pos_weight: torch.Tensor | None = None,
        tox_pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        if bio_pos_weight is not None and bio_pos_weight.shape != (N_BIO,):
            raise ValueError(
                f"bio_pos_weight: expected shape {(N_BIO,)}, got "
                f"{tuple(bio_pos_weight.shape)}"
            )
        if tox_pos_weight is not None and tox_pos_weight.shape != (N_TOX,):
            raise ValueError(
                f"tox_pos_weight: expected shape {(N_TOX,)}, got "
                f"{tuple(tox_pos_weight.shape)}"
            )
        self.register_buffer(
            "bio_pos_weight",
            bio_pos_weight.float() if bio_pos_weight is not None else None,
        )
        self.register_buffer(
            "tox_pos_weight",
            tox_pos_weight.float() if tox_pos_weight is not None else None,
        )

    @classmethod
    def from_dataset(cls, dataset: MuFuDataset, pos_weight: str = "raw") -> "TaskLoss":
        weights = dataset_pos_weights(dataset, mode=pos_weight)
        return cls(
            bio_pos_weight=weights["bio_cls"],
            tox_pos_weight=weights["tox"],
        )

    def forward(self, outputs: MuFuOutputs, batch: MuFuBatch) -> TaskLosses:
        return compute_task_losses(
            outputs,
            batch,
            pos_weights={
                "bio_cls": self.bio_pos_weight,
                "tox": self.tox_pos_weight,
            },
        )
