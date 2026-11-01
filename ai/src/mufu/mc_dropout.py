"""
MC-Dropout spread and conformal intervals for MuFu's pIC50 predictions.

`mc_predict` takes the point estimate from the normal eval-mode model and sigma from
`n_samples` dropout-on passes of `StochasticMuFu`. `fit_conformal` calibrates a
multiplier `q` per target and coverage, giving `pred +/- q * max(sigma, floor)`.

`q` is only valid for the checkpoint, dropout rates and `n_samples` it was fitted with.
It is saved next to the checkpoint as `<stem>_mc_dropout.json`.
"""

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mufu.contract import N_BIO, PIC50_COLUMNS
from mufu.data.dataset import MuFuDataset, collate, to_device
from mufu.model import MuFu

DEFAULT_T = 50
DEFAULT_COVERAGES = (0.9, 0.95)
SIGMA_FLOOR_PERCENTILE = 5.0
SIGMA_FLOOR_MIN = 1e-6
SIDECAR_SUFFIX = "_mc_dropout.json"
PIC50_PARTS = (
    "fingerprint_encoder",
    "graph_encoder",
    "projection",
    "fusion",
    "regression_head",
)


class MCDropout(nn.Module):
    """
    Dropout that stays on in eval mode.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True)


def _swap_dropout(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, MCDropout(child.p))
        else:
            _swap_dropout(child)


class StochasticMuFu(nn.Module):
    """
    MuFu's pIC50 path with dropout always on minus frozen MolFormer encoder.

    Takes precomputed MolFormer embedding alongside fingerprint and graph
    inputs. Normalization layers stay in eval mode and fusion gets no modality mask.
    """

    def __init__(self, model: MuFu) -> None:
        super().__init__()
        for name in PIC50_PARTS:
            setattr(self, name, copy.deepcopy(getattr(model, name)))
        _swap_dropout(self)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @property
    def dropout_rates(self) -> tuple[float, ...]:
        return tuple(m.p for m in self.modules() if isinstance(m, MCDropout))

    def forward(
        self, smiles_embedding: torch.Tensor, fingerprint: torch.Tensor, graph
    ) -> torch.Tensor:
        projected = self.projection(
            {
                "smiles": smiles_embedding,
                "fingerprint": self.fingerprint_encoder(fingerprint),
                "graph": self.graph_encoder(graph),
            }
        )
        return self.regression_head(self.fusion(projected, None))


@torch.no_grad()
def mc_predict(
    model: MuFu,
    dataset: MuFuDataset,
    *,
    n_samples: int = DEFAULT_T,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return eval-mode pIC50 and standard deviation of `n_samples` MC-Dropout passes.
    SMILES encoder (MolFormer) runs once per batch.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    model.eval()
    stochastic = StochasticMuFu(model)
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    preds, sigmas = [], []
    for batch in loader:
        batch = to_device(batch, device)
        embedding = model.smiles_encoder(batch["smiles"])
        preds.append(model.forward_embedded(embedding, batch)["pic50"].cpu())
        samples = torch.stack(
            [
                stochastic(embedding, batch["fingerprint"], batch["graph"])
                for _ in range(n_samples)
            ]
        )
        sigmas.append(samples.std(dim=0).cpu())
    return torch.cat(preds).numpy(), torch.cat(sigmas).numpy()


@dataclass(frozen=True)
class ConformalParams:
    coverages: tuple[float, ...]
    q: np.ndarray  # (N_BIO, len(coverages))
    sigma_floor: np.ndarray  # (N_BIO,)
    n_samples: int
    dropout_rates: tuple[float, ...]
    checkpoint_sha256: str = ""

    def intervals(
        self, pred: np.ndarray, sigma: np.ndarray
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        scale = np.maximum(sigma, self.sigma_floor)
        return {
            cov: (pred - self.q[:, i] * scale, pred + self.q[:, i] * scale)
            for i, cov in enumerate(self.coverages)
        }

    def to_json(self) -> str:
        state = asdict(self)
        state["q"] = self.q.tolist()
        state["sigma_floor"] = self.sigma_floor.tolist()
        return json.dumps(state, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ConformalParams":
        state = json.loads(text)
        return cls(
            **{
                **state,
                "coverages": tuple(state["coverages"]),
                "q": np.array(state["q"]),
                "sigma_floor": np.array(state["sigma_floor"]),
                "dropout_rates": tuple(state["dropout_rates"]),
            }
        )


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """
    ceil((n+1) * coverage)-th smallest score, or inf if there are too few scores.
    """
    n = len(scores)
    k = math.ceil(round((n + 1) * coverage, 9))
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def fit_conformal(
    pred: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    n_samples: int,
    dropout_rates: tuple[float, ...],
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> ConformalParams:
    """
    Fit `q` per target and coverage from |y - pred| / max(sigma, floor) on labeled rows.
    The floor is a low percentile of that target's sigma.
    """
    q = np.empty((N_BIO, len(coverages)))
    floor = np.empty(N_BIO)
    for j in range(N_BIO):
        valid = mask[:, j]
        if not valid.any():
            raise ValueError(f"{PIC50_COLUMNS[j]}: no labeled rows to calibrate on")
        floor[j] = max(
            np.percentile(sigma[valid, j], SIGMA_FLOOR_PERCENTILE), SIGMA_FLOOR_MIN
        )
        scores = np.abs(y[valid, j] - pred[valid, j]) / np.maximum(
            sigma[valid, j], floor[j]
        )
        q[j] = [conformal_quantile(scores, cov) for cov in coverages]
    return ConformalParams(tuple(coverages), q, floor, n_samples, dropout_rates)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_name(ckpt_path.stem + SIDECAR_SUFFIX)


def save_conformal(params: ConformalParams, ckpt_path: Path) -> None:
    params = replace(params, checkpoint_sha256=_sha256(ckpt_path))
    sidecar_path(ckpt_path).write_text(params.to_json() + "\n")


def load_conformal(ckpt_path: Path) -> ConformalParams:
    params = ConformalParams.from_json(sidecar_path(ckpt_path).read_text())
    if params.checkpoint_sha256 != _sha256(ckpt_path):
        raise ValueError(
            f"{ckpt_path.name}: saved q was fitted on a different checkpoint"
        )
    return params


def fit_mc_dropout(
    model: MuFu,
    calibration_dataset: MuFuDataset,
    ckpt_path: Path,
    *,
    n_samples: int = DEFAULT_T,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
    batch_size: int = 64,
) -> ConformalParams:
    """
    Fit `q` on the calibration split and save it next to the checkpoint.
    """
    pred, sigma = mc_predict(
        model, calibration_dataset, n_samples=n_samples, batch_size=batch_size
    )
    params = fit_conformal(
        pred,
        sigma,
        calibration_dataset.labels["y_pic50"].numpy(),
        calibration_dataset.masks["mask_bio"].numpy(),
        n_samples=n_samples,
        dropout_rates=StochasticMuFu(model).dropout_rates,
        coverages=coverages,
    )
    save_conformal(params, ckpt_path)
    return params


def predict_with_intervals(
    model: MuFu,
    dataset: MuFuDataset,
    conformal: ConformalParams,
    *,
    batch_size: int = 64,
) -> dict[str, list[str] | np.ndarray]:
    """
    Per target: `<col>`, `<col>_sigma`, and `<col>_lo<percent>` / `<col>_hi<percent>`
    for each coverage. Raises if model's dropout rates differ from the ones `q` was
    fitted with.
    """
    if StochasticMuFu(model).dropout_rates != conformal.dropout_rates:
        raise ValueError("dropout rates differ from those q was fitted with; refit q")

    pred, sigma = mc_predict(
        model, dataset, n_samples=conformal.n_samples, batch_size=batch_size
    )
    out: dict[str, list[str] | np.ndarray] = {"smiles": list(dataset.smiles)}
    bounds = conformal.intervals(pred, sigma)
    for j, col in enumerate(PIC50_COLUMNS):
        out[col] = pred[:, j]
        out[f"{col}_sigma"] = sigma[:, j]
        for cov, (lo, hi) in bounds.items():
            pct = round(cov * 100)
            out[f"{col}_lo{pct}"] = lo[:, j]
            out[f"{col}_hi{pct}"] = hi[:, j]
    return out


def empirical_coverage(
    preds: dict[str, list[str] | np.ndarray],
    dataset: MuFuDataset,
    coverages: tuple[float, ...],
) -> pd.DataFrame:
    """
    Share of labeled compounds inside their interval and mean interval width, per
    target and coverage.
    """
    y = dataset.labels["y_pic50"].numpy()
    mask = dataset.masks["mask_bio"].numpy()
    rows = []
    for j, col in enumerate(PIC50_COLUMNS):
        valid = mask[:, j]
        for cov in coverages:
            pct = round(cov * 100)
            lo, hi = preds[f"{col}_lo{pct}"][valid], preds[f"{col}_hi{pct}"][valid]
            inside = (y[valid, j] >= lo) & (y[valid, j] <= hi)
            rows.append(
                {
                    "target": col,
                    "nominal": cov,
                    "empirical": float(inside.mean()),
                    "n": int(valid.sum()),
                    "mean_width": float((hi - lo).mean()),
                }
            )
    return pd.DataFrame(rows)
