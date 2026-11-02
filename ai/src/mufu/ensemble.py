"""
Seed ensemble of MuFu models with MC-Dropout intervals for pIC50.

Probabilities are averaged after each member's own calibration: member s applies its
Platt scale and shift to its own logits and the sigmoids are averaged. pIC50 is the mean
of the members' eval-mode predictions.

The interval combines MC-Dropout and the spread between members,

    sigma^2 = mean_s(sigma_s^2) + sum_s(mu_s - mean(mu))^2 / (S - 1),

with sigma_s  being the `n_samples` MC-Dropout standard deviation of member s. The
interval is `mu +/- q * max(sigma, floor)` with `q` and the floor fitted on val.

`q` is only valid for the members `n_samples` and dropout rates it was fitted with. It's
saved as `<model_name>_ensemble.json` next to checkpoints with each member's sha256.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from baseline.results import file_sha256
from mufu.calibration import CalibrationParams
from mufu.contract import PIC50_COLUMNS
from mufu.data.dataset import MuFuDataset
from mufu.experiment import load_mufu_checkpoint
from mufu.mc_dropout import (
    DEFAULT_COVERAGES,
    DEFAULT_T,
    ConformalParams,
    StochasticMuFu,
    fit_conformal,
    mc_predict,
)
from mufu.model import MuFu
from mufu.train import predict

SIDECAR_SUFFIX = "_ensemble.json"


@dataclass(frozen=True)
class Member:
    path: Path
    model: MuFu
    family_states: Mapping[str, Mapping[str, torch.Tensor]] | None
    calibration: CalibrationParams | None


@dataclass(frozen=True)
class EnsembleParams:
    conformal: ConformalParams
    members: tuple[tuple[str, str], ...]  # (checkpoint file name, sha256)


def member_paths(
    checkpoint_dir: Path, model_name: str, seeds: Sequence[int]
) -> list[Path]:
    paths = [checkpoint_dir / f"{model_name}_seed_{seed}.pt" for seed in seeds]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {', '.join(missing)}")
    return paths


def load_members(
    paths: Sequence[Path],
    smiles_encoder,
    *,
    device: torch.device | str = "cpu",
) -> list[Member]:
    members, fingerprints, seeds = [], [], []
    for path in paths:
        model, meta = load_mufu_checkpoint(path, smiles_encoder, device=device)
        ckpt = meta["checkpoint"]
        fingerprints.append((ckpt.get("params"), ckpt.get("data_sha256")))
        seeds.append(ckpt.get("seed"))
        members.append(Member(path, model, meta["family_states"], meta["calibration"]))
    if any(f != fingerprints[0] for f in fingerprints):
        raise ValueError("members have different params or data_sha256")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"members share a seed: {seeds}")
    return members


def average_predictions(member_preds: Sequence[Mapping]) -> dict:
    """
    Mean of every non-SMILES column across members, checking that SMILES match.
    """
    smiles = list(member_preds[0]["smiles"])
    if any(list(p["smiles"]) != smiles for p in member_preds):
        raise ValueError("members predict different compounds or a different order")
    return {
        "smiles": smiles,
        **{
            col: np.mean([np.asarray(p[col], dtype=float) for p in member_preds], 0)
            for col in member_preds[0]
            if col != "smiles"
        },
    }


def combined_sigma(mu: np.ndarray, mc_sigma: np.ndarray) -> np.ndarray:
    """
    sqrt(mean of MC-Dropout variances + sample variance of member means), both
    shaped (S, N, N_BIO).
    """
    return np.sqrt((mc_sigma**2).mean(axis=0) + mu.var(axis=0, ddof=1))


def ensemble_predict(
    members: Sequence[Member],
    dataset: MuFuDataset,
    *,
    n_samples: int = DEFAULT_T,
    batch_size: int = 64,
) -> tuple[dict, np.ndarray]:
    """
    Return `predict`-shaped averaged predictions and the combined pIC50 sigma
    (N, N_BIO).
    """
    if len(members) < 2:
        raise ValueError("an ensemble needs at least 2 members")
    preds, mc_sigma = [], []
    for m in members:
        preds.append(
            predict(
                m.model,
                dataset,
                batch_size=batch_size,
                family_states=m.family_states,
                calibration=m.calibration,
            )
        )
        mc_sigma.append(
            mc_predict(m.model, dataset, n_samples=n_samples, batch_size=batch_size)[1]
        )
    mean = average_predictions(preds)
    mu = np.stack([np.column_stack([p[c] for c in PIC50_COLUMNS]) for p in preds])
    return mean, combined_sigma(mu, np.stack(mc_sigma))


def _dropout_rates(members: Sequence[Member]) -> tuple[float, ...]:
    rates = {StochasticMuFu(m.model).dropout_rates for m in members}
    if len(rates) != 1:
        raise ValueError("members have different dropout rates")
    return rates.pop()


def fit_ensemble_conformal(
    members: Sequence[Member],
    val_dataset: MuFuDataset,
    *,
    n_samples: int = DEFAULT_T,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
    batch_size: int = 64,
) -> EnsembleParams:
    torch.manual_seed(0)
    rates = _dropout_rates(members)
    mean, sigma = ensemble_predict(
        members, val_dataset, n_samples=n_samples, batch_size=batch_size
    )
    conformal = fit_conformal(
        np.column_stack([mean[c] for c in PIC50_COLUMNS]),
        sigma,
        val_dataset.labels["y_pic50"].numpy(),
        val_dataset.masks["mask_bio"].numpy(),
        n_samples=n_samples,
        dropout_rates=rates,
        coverages=coverages,
    )
    return EnsembleParams(
        conformal, tuple((m.path.name, file_sha256(m.path)) for m in members)
    )


def sidecar_path(checkpoint_dir: Path, model_name: str) -> Path:
    return checkpoint_dir / f"{model_name}{SIDECAR_SUFFIX}"


def save_ensemble(
    params: EnsembleParams, checkpoint_dir: Path, model_name: str
) -> None:
    state = {
        "members": [{"checkpoint": n, "sha256": s} for n, s in params.members],
        "conformal": json.loads(params.conformal.to_json()),
    }
    sidecar_path(checkpoint_dir, model_name).write_text(
        json.dumps(state, indent=2) + "\n"
    )


def load_ensemble_params(
    paths: Sequence[Path], checkpoint_dir: Path, model_name: str
) -> EnsembleParams:
    state = json.loads(sidecar_path(checkpoint_dir, model_name).read_text())
    saved = tuple((m["checkpoint"], m["sha256"]) for m in state["members"])
    current = tuple((p.name, file_sha256(p)) for p in paths)
    if saved != current:
        raise ValueError("saved q was fitted on different members; refit the ensemble")
    return EnsembleParams(
        ConformalParams.from_json(json.dumps(state["conformal"])), saved
    )


def predict_with_ensemble_intervals(
    members: Sequence[Member],
    dataset: MuFuDataset,
    params: EnsembleParams,
    *,
    batch_size: int = 64,
) -> dict[str, list[str] | np.ndarray]:
    """
    Same keys as `predict_with_intervals` (`<col>`, `<col>_sigma`, `<col>_lo<percent>`,
    `<col>_hi<percent>`) plus averaged probabilities. Raises if a member's dropout rates
    differ from the ones `q` was fitted with.
    """
    conformal = params.conformal
    if _dropout_rates(members) != conformal.dropout_rates:
        raise ValueError("dropout rates differ from those q was fitted with; refit q")

    torch.manual_seed(0)
    mean, sigma = ensemble_predict(
        members, dataset, n_samples=conformal.n_samples, batch_size=batch_size
    )
    pred = np.column_stack([mean[c] for c in PIC50_COLUMNS])
    bounds = conformal.intervals(pred, sigma)
    out: dict[str, list[str] | np.ndarray] = dict(mean)
    for j, col in enumerate(PIC50_COLUMNS):
        out[f"{col}_sigma"] = sigma[:, j]
        for cov, (lo, hi) in bounds.items():
            pct = round(cov * 100)
            out[f"{col}_lo{pct}"] = lo[:, j]
            out[f"{col}_hi{pct}"] = hi[:, j]
    return out
