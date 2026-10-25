"""
Post-hoc logit calibration and calibration metrics for MuFu classification heads.

Calibration transforms classification logits via a positive per-column scale and shift
(`scale * logits + shift`) before the sigmoid, preserving per-column ranking (ROC-AUC)
while correcting probability bias and overconfidence from `pos_weight` and overfitting.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mufu.contract import ACTIVE_COLUMNS, N_BIO, N_TOX, TOX_COLUMNS
from mufu.data.dataset import MuFuDataset
from pipeline.toxicity_loader import DATASETS

CALIBRATION_MODES = ("none", "log_weight", "temperature", "platt")
LOG_SCALE_MIN = -3.0
LOG_SCALE_MAX = 3.0


@dataclass(frozen=True)
class CalibrationParams:
    active_scale: torch.Tensor
    active_shift: torch.Tensor
    tox_scale: torch.Tensor
    tox_shift: torch.Tensor

    def __post_init__(self) -> None:
        if self.active_scale.shape != (N_BIO,) or self.active_shift.shape != (N_BIO,):
            raise ValueError(
                f"active scale/shift must have shape {(N_BIO,)}, got "
                f"{tuple(self.active_scale.shape)} and {tuple(self.active_shift.shape)}"
            )
        if self.tox_scale.shape != (N_TOX,) or self.tox_shift.shape != (N_TOX,):
            raise ValueError(
                f"tox scale/shift must have shape {(N_TOX,)}, got "
                f"{tuple(self.tox_scale.shape)} and {tuple(self.tox_shift.shape)}"
            )
        if (self.active_scale <= 0).any() or (self.tox_scale <= 0).any():
            raise ValueError("calibration scales must be strictly positive")

    @classmethod
    def identity(cls) -> "CalibrationParams":
        return cls(
            active_scale=torch.ones(N_BIO, dtype=torch.float32),
            active_shift=torch.zeros(N_BIO, dtype=torch.float32),
            tox_scale=torch.ones(N_TOX, dtype=torch.float32),
            tox_shift=torch.zeros(N_TOX, dtype=torch.float32),
        )

    def apply(self, outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(outputs)
        for key, scale, shift in (
            ("active_logits", self.active_scale, self.active_shift),
            ("tox_logits", self.tox_scale, self.tox_shift),
        ):
            if key in out:
                logits = out[key]
                s = scale.to(device=logits.device, dtype=logits.dtype)
                b = shift.to(device=logits.device, dtype=logits.dtype)
                out[key] = logits * s + b
        return out

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "active_scale": self.active_scale.detach().cpu().clone(),
            "active_shift": self.active_shift.detach().cpu().clone(),
            "tox_scale": self.tox_scale.detach().cpu().clone(),
            "tox_shift": self.tox_shift.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> "CalibrationParams":
        return cls(
            active_scale=state["active_scale"].detach().cpu().float().clone(),
            active_shift=state["active_shift"].detach().cpu().float().clone(),
            tox_scale=state["tox_scale"].detach().cpu().float().clone(),
            tox_shift=state["tox_shift"].detach().cpu().float().clone(),
        )


def _prior_shift(pos_weight: torch.Tensor | None, width: int) -> torch.Tensor:
    if pos_weight is None:
        return torch.zeros(width, dtype=torch.float32)
    pw = pos_weight.detach().cpu().float()
    if pw.shape != (width,):
        raise ValueError(
            f"pos_weight: expected shape {(width,)}, got {tuple(pw.shape)}"
        )
    return -torch.log(pw)


def _fit_temperature_column(
    logits: torch.Tensor, labels: torch.Tensor, base_shift: float
) -> tuple[float, float]:
    shifted = (logits.detach().cpu().float() + base_shift).double()
    y = labels.detach().cpu().double()
    if y.numel() == 0 or torch.unique(y).numel() < 2:
        return 1.0, float(base_shift)

    # Grid search over log_inv_temp in [LOG_SCALE_MIN, LOG_SCALE_MAX] then polish
    grid = torch.linspace(LOG_SCALE_MIN, LOG_SCALE_MAX, 121, dtype=torch.double)
    scales = torch.exp(grid)
    losses = F.binary_cross_entropy_with_logits(
        shifted[:, None] * scales[None, :],
        y[:, None].expand(-1, scales.numel()),
        reduction="none",
    ).mean(dim=0)
    best_log_scale = grid[int(torch.argmin(losses).item())].clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [best_log_scale], lr=0.5, max_iter=25, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        clamped = best_log_scale.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
        scale = torch.exp(clamped)
        loss = F.binary_cross_entropy_with_logits(shifted * scale, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_scale = float(
        torch.exp(best_log_scale.detach().clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)).item()
    )
    if not np.isfinite(final_scale) or final_scale <= 0:
        return 1.0, float(base_shift)
    return final_scale, float(base_shift * final_scale)


def _fit_platt_column(
    logits: torch.Tensor, labels: torch.Tensor, base_shift: float
) -> tuple[float, float]:
    z = logits.detach().cpu().double()
    y = labels.detach().cpu().double()
    if z.numel() == 0 or torch.unique(y).numel() < 2:
        return 1.0, float(base_shift)

    log_scale = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    shift = torch.tensor(float(base_shift), dtype=torch.double, requires_grad=True)
    base_loss = float(F.binary_cross_entropy_with_logits(z + base_shift, y).item())

    optimizer = torch.optim.LBFGS(
        [log_scale, shift], lr=0.5, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        clamped = log_scale.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
        scale = torch.exp(clamped)
        reg = 1e-4 * (clamped.pow(2) + (shift - base_shift).pow(2))
        loss = F.binary_cross_entropy_with_logits(z * scale + shift, y) + reg
        loss.backward()
        return loss

    optimizer.step(closure)
    final_scale = float(
        torch.exp(log_scale.detach().clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)).item()
    )
    final_shift = float(shift.detach().item())
    final_loss = float(
        F.binary_cross_entropy_with_logits(z * final_scale + final_shift, y).item()
    )
    if (
        not np.isfinite(final_scale)
        or not np.isfinite(final_shift)
        or final_scale <= 0
        or final_loss > base_loss + 1e-5
    ):
        return 1.0, float(base_shift)
    return final_scale, final_shift


def _fit_head_columns(
    mode: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    base_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = logits.shape[1]
    scales = torch.ones(width, dtype=torch.float32)
    shifts = base_shifts.clone()

    for j in range(width):
        valid = mask[:, j]
        col_logits = logits[valid, j]
        col_labels = labels[valid, j]
        b0 = float(base_shifts[j].item())
        if mode == "temperature":
            s, b = _fit_temperature_column(col_logits, col_labels, b0)
        else:
            s, b = _fit_platt_column(col_logits, col_labels, b0)
        scales[j] = s
        shifts[j] = b
    return scales, shifts


def fit_calibration(
    mode: str,
    val_logits: Mapping[str, torch.Tensor],
    val_dataset: MuFuDataset,
    pos_weights: Mapping[str, torch.Tensor | None] | None = None,
) -> CalibrationParams:
    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration {mode!r}; expected one of {CALIBRATION_MODES}"
        )
    if mode == "none":
        return CalibrationParams.identity()

    weights = pos_weights or {}
    active_base = _prior_shift(weights.get("bio_cls"), N_BIO)
    tox_base = _prior_shift(weights.get("tox"), N_TOX)

    if mode == "log_weight":
        return CalibrationParams(
            active_scale=torch.ones(N_BIO, dtype=torch.float32),
            active_shift=active_base,
            tox_scale=torch.ones(N_TOX, dtype=torch.float32),
            tox_shift=tox_base,
        )

    active_scale, active_shift = _fit_head_columns(
        mode,
        val_logits["active_logits"].detach().cpu(),
        val_dataset.labels["y_active"].detach().cpu(),
        val_dataset.masks["mask_bio"].detach().cpu(),
        active_base,
    )
    tox_scale, tox_shift = _fit_head_columns(
        mode,
        val_logits["tox_logits"].detach().cpu(),
        val_dataset.labels["y_tox"].detach().cpu(),
        val_dataset.masks["mask_tox"].detach().cpu(),
        tox_base,
    )
    return CalibrationParams(
        active_scale=active_scale,
        active_shift=active_shift,
        tox_scale=tox_scale,
        tox_shift=tox_shift,
    )


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        in_bin = (
            (probs >= low) & (probs <= high)
            if i == n_bins - 1
            else (probs >= low) & (probs < high)
        )
        count = int(in_bin.sum())
        if count > 0:
            ece += (count / n) * abs(
                float(probs[in_bin].mean()) - float(labels[in_bin].mean())
            )
    return float(ece)


def evaluate_calibration(
    probs: Mapping[str, Sequence[float] | np.ndarray],
    dataset: MuFuDataset,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []
    tox21_tasks = set(DATASETS["tox21"]["tasks"])
    clintox_tasks = set(DATASETS["clintox"]["tasks"])

    groups = (
        ("bio_cls", ACTIVE_COLUMNS, "y_active", "mask_bio"),
        ("tox", TOX_COLUMNS, "y_tox", "mask_tox"),
    )
    for family, columns, label_key, mask_key in groups:
        y_mat = dataset.labels[label_key].detach().cpu().numpy()
        m_mat = dataset.masks[mask_key].detach().cpu().numpy()
        for j, col in enumerate(columns):
            if col not in probs:
                continue
            valid = m_mat[:, j]
            n_labeled = int(valid.sum())
            if n_labeled == 0:
                continue
            y = y_mat[valid, j].astype(float)
            p = np.asarray(probs[col], dtype=float)[valid]
            clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
            prevalence = float(y.mean())
            mean_prob = float(p.mean())
            bias = mean_prob - prevalence
            brier = float(np.mean((p - y) ** 2))
            bce = float(
                -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
            )
            ece = expected_calibration_error(p, y, n_bins=n_bins)
            if family == "bio_cls":
                group = "bio_cls"
            elif col in tox21_tasks:
                group = "tox21"
            elif col in clintox_tasks:
                group = "clintox"
            else:
                group = "tox"
            rows.append(
                {
                    "group": group,
                    "endpoint": col,
                    "n_labeled": n_labeled,
                    "n_positive": int(np.sum(y == 1.0)),
                    "prevalence": prevalence,
                    "mean_prob": mean_prob,
                    "bias": bias,
                    "abs_bias": abs(bias),
                    "brier": brier,
                    "bce": bce,
                    "ece": ece,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    metric_cols = ["prevalence", "mean_prob", "bias", "abs_bias", "brier", "bce", "ece"]
    summaries: list[dict] = []
    for name, subset in (
        ("macro_bio_cls", df[df["group"] == "bio_cls"]),
        ("macro_tox21", df[df["group"] == "tox21"]),
        ("macro_clintox", df[df["group"] == "clintox"]),
        ("macro_tox", df[df["group"].isin(["tox21", "clintox"])]),
    ):
        if subset.empty:
            continue
        summaries.append(
            {
                "group": "summary",
                "endpoint": name,
                "n_labeled": int(subset["n_labeled"].sum()),
                "n_positive": int(subset["n_positive"].sum()),
                **{c: float(subset[c].mean()) for c in metric_cols},
            }
        )
    return pd.concat([df, pd.DataFrame(summaries)], ignore_index=True)
