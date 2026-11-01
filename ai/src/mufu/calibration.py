"""
Post-hoc calibration and calibration metrics for MuFu's classification heads.

Each column's logits become `scale * logits + shift` before the sigmoid. `scale` is
kept positive, so ROC-AUC doesn't change.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

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
HEADS = (("active", "active_logits", N_BIO), ("tox", "tox_logits", N_TOX))


@dataclass(frozen=True)
class CalibrationParams:
    active_scale: torch.Tensor
    active_shift: torch.Tensor
    tox_scale: torch.Tensor
    tox_shift: torch.Tensor

    def __post_init__(self) -> None:
        for head, _, width in HEADS:
            for part in ("scale", "shift"):
                shape = tuple(getattr(self, f"{head}_{part}").shape)
                if shape != (width,):
                    raise ValueError(
                        f"{head}_{part}: expected shape {(width,)}, got {shape}"
                    )
        if (self.active_scale <= 0).any() or (self.tox_scale <= 0).any():
            raise ValueError("calibration scales must be strictly positive")

    @classmethod
    def identity(cls) -> "CalibrationParams":
        return cls(
            active_scale=torch.ones(N_BIO),
            active_shift=torch.zeros(N_BIO),
            tox_scale=torch.ones(N_TOX),
            tox_shift=torch.zeros(N_TOX),
        )

    def apply(self, outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(outputs)
        for head, key, _ in HEADS:
            if key in out:
                logits = out[key]
                scale = getattr(self, f"{head}_scale").to(logits)
                shift = getattr(self, f"{head}_shift").to(logits)
                out[key] = logits * scale + shift
        return out

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in asdict(self).items()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> "CalibrationParams":
        return cls(**{k: v.detach().cpu().float().clone() for k, v in state.items()})


def _prior_shift(pos_weight: torch.Tensor | None, width: int) -> torch.Tensor:
    if pos_weight is None:
        return torch.zeros(width)
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

    # grid search for a starting point, then polish with L-BFGS
    grid = torch.linspace(LOG_SCALE_MIN, LOG_SCALE_MAX, 121, dtype=torch.double)
    scales = torch.exp(grid)
    losses = F.binary_cross_entropy_with_logits(
        shifted[:, None] * scales[None, :],
        y[:, None].expand(-1, scales.numel()),
        reduction="none",
    ).mean(dim=0)
    log_scale = grid[int(torch.argmin(losses).item())].clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [log_scale], lr=0.5, max_iter=25, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = torch.exp(log_scale.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX))
        loss = F.binary_cross_entropy_with_logits(shifted * scale, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(torch.exp(log_scale.detach().clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0, float(base_shift)
    return scale, float(base_shift * scale)


def _fit_platt_column(
    logits: torch.Tensor, labels: torch.Tensor, base_shift: float
) -> tuple[float, float]:
    z = logits.detach().cpu().double()
    y = labels.detach().cpu().double()
    if z.numel() == 0 or torch.unique(y).numel() < 2:
        return 1.0, float(base_shift)

    log_scale = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    shift = torch.tensor(float(base_shift), dtype=torch.double, requires_grad=True)
    base_loss = float(F.binary_cross_entropy_with_logits(z + base_shift, y))

    optimizer = torch.optim.LBFGS(
        [log_scale, shift], lr=0.5, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        clamped = log_scale.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
        reg = 1e-4 * (clamped.pow(2) + (shift - base_shift).pow(2))
        loss = (
            F.binary_cross_entropy_with_logits(z * torch.exp(clamped) + shift, y) + reg
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(torch.exp(log_scale.detach().clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)))
    b = float(shift.detach())
    loss = float(F.binary_cross_entropy_with_logits(z * scale + b, y))
    # fall back to the prior shift if the fit diverged or didn't beat it
    if (
        not (np.isfinite(scale) and np.isfinite(b) and scale > 0)
        or loss > base_loss + 1e-5
    ):
        return 1.0, float(base_shift)
    return scale, b


def fit_calibration(
    mode: str,
    val_logits: Mapping[str, torch.Tensor],
    val_dataset: MuFuDataset,
    pos_weights: Mapping[str, torch.Tensor | None] | None = None,
) -> CalibrationParams:
    """
    Fit per-column calibration on validation logits.

    `log_weight` only undoes the `pos_weight` prior shift, `-log(pos_weight)`.
    `temperature` and `platt` start from that shift and fit a scale (and, for `platt`,
    a shift) to minimize BCE without `pos_weight`.
    """
    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration {mode!r}; expected one of {CALIBRATION_MODES}"
        )
    if mode == "none":
        return CalibrationParams.identity()

    weights = pos_weights or {}
    fit_column = _fit_temperature_column if mode == "temperature" else _fit_platt_column
    params = {}
    for (head, key, width), (loss_key, label_key, mask_key) in zip(
        HEADS, (("bio_cls", "y_active", "mask_bio"), ("tox", "y_tox", "mask_tox"))
    ):
        base = _prior_shift(weights.get(loss_key), width)
        scales, shifts = torch.ones(width), base.clone()
        if mode != "log_weight":
            logits = val_logits[key].detach().cpu()
            labels = val_dataset.labels[label_key].detach().cpu()
            mask = val_dataset.masks[mask_key].detach().cpu()
            for j in range(width):
                valid = mask[:, j]
                scales[j], shifts[j] = fit_column(
                    logits[valid, j], labels[valid, j], float(base[j])
                )
        params[f"{head}_scale"], params[f"{head}_shift"] = scales, shifts
    return CalibrationParams(**params)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    if len(probs) == 0:
        return float("nan")
    bins = np.digitize(probs, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return float(
        sum(
            (bins == b).mean() * abs(probs[bins == b].mean() - labels[bins == b].mean())
            for b in np.unique(bins)
        )
    )


def evaluate_calibration(
    probs: Mapping[str, Sequence[float] | np.ndarray],
    dataset: MuFuDataset,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Per-endpoint and macro calibration metrics: mean predicted probability vs
    prevalence, bias, Brier score, BCE, and ECE.
    """
    groups = {
        **dict.fromkeys(ACTIVE_COLUMNS, "bio_cls"),
        **dict.fromkeys(DATASETS["tox21"]["tasks"], "tox21"),
        **dict.fromkeys(DATASETS["clintox"]["tasks"], "clintox"),
    }
    rows: list[dict] = []
    for columns, label_key, mask_key in (
        (ACTIVE_COLUMNS, "y_active", "mask_bio"),
        (TOX_COLUMNS, "y_tox", "mask_tox"),
    ):
        y_mat = dataset.labels[label_key].detach().cpu().numpy()
        m_mat = dataset.masks[mask_key].detach().cpu().numpy()
        for j, col in enumerate(columns):
            valid = m_mat[:, j]
            if col not in probs or not valid.any():
                continue
            y = y_mat[valid, j].astype(float)
            p = np.asarray(probs[col], dtype=float)[valid]
            clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
            bias = float(p.mean()) - float(y.mean())
            rows.append(
                {
                    "group": groups.get(col, "tox"),
                    "endpoint": col,
                    "n_labeled": int(valid.sum()),
                    "n_positive": int(np.sum(y == 1.0)),
                    "prevalence": float(y.mean()),
                    "mean_prob": float(p.mean()),
                    "bias": bias,
                    "abs_bias": abs(bias),
                    "brier": float(np.mean((p - y) ** 2)),
                    "bce": float(
                        -np.mean(
                            y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)
                        )
                    ),
                    "ece": expected_calibration_error(p, y, n_bins=n_bins),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    metric_cols = ["prevalence", "mean_prob", "bias", "abs_bias", "brier", "bce", "ece"]
    summaries = [
        {
            "group": "summary",
            "endpoint": name,
            "n_labeled": int(subset["n_labeled"].sum()),
            "n_positive": int(subset["n_positive"].sum()),
            **{c: float(subset[c].mean()) for c in metric_cols},
        }
        for name, subset in (
            ("macro_bio_cls", df[df["group"] == "bio_cls"]),
            ("macro_tox21", df[df["group"] == "tox21"]),
            ("macro_clintox", df[df["group"] == "clintox"]),
            ("macro_tox", df[df["group"].isin(["tox21", "clintox"])]),
        )
        if not subset.empty
    ]
    return pd.concat([df, pd.DataFrame(summaries)], ignore_index=True)
