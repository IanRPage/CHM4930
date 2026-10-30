"""
MuFu 5-seed ensemble inference for one molecule, in the response format of
`backend/docs/predict-api.md`.

pIC50 is the ensemble mean with its combined MC-Dropout interval at `COVERAGE`;
`p_active` and toxicity average each seed's calibrated probabilities. MC passes are
seeded, so the same SMILES always gets the same response.
"""

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
import torch

from baseline.benchmark import SEEDS
from baseline.results import file_sha256
from encoders.smiles_encoder import SmilesEncoder
from mufu.contract import ACTIVE_COLUMNS, BIO_TARGETS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import MuFuDataset
from mufu.ensemble import (
    EnsembleParams,
    Member,
    load_ensemble_params,
    load_members,
    member_paths,
    predict_with_ensemble_intervals,
    sidecar_path,
)
from mufu.hub import download_version, load_tagged_version
from mufu_server.config import COVERAGE, Settings

MC_SEED = 0


class InvalidSmiles(ValueError):
    pass


@dataclass
class Predictor:
    members: Sequence[Member]
    params: EnsembleParams
    model_version: str
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("an ensemble needs at least 2 members")
        if COVERAGE not in self.params.conformal.coverages:
            raise ValueError(f"ensemble params have no {COVERAGE} coverage")

    def infer(self, smiles: str) -> dict:
        frame = pd.DataFrame({"smiles": [smiles]})
        for column in (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS):
            frame[column] = float("nan")
        dataset = MuFuDataset(frame)
        if len(dataset) == 0:
            raise InvalidSmiles(smiles)

        # RNG and encoder's embedding cache shared across threads
        with self._lock, torch.random.fork_rng(devices=[]):
            torch.manual_seed(MC_SEED)
            preds = predict_with_ensemble_intervals(self.members, dataset, self.params)

        pct = round(COVERAGE * 100)
        bioactivity = {}
        for target, pic50_col, active_col in zip(
            BIO_TARGETS, PIC50_COLUMNS, ACTIVE_COLUMNS
        ):
            value, lo, hi = (
                _finite(preds[col][0])
                for col in (pic50_col, f"{pic50_col}_lo{pct}", f"{pic50_col}_hi{pct}")
            )
            if None in (value, lo, hi):
                value = lo = hi = None
            bioactivity[target] = {
                "pic50": value,
                "pic50_lo": lo,
                "pic50_hi": hi,
                "p_active": _finite(preds[active_col][0]),
            }

        response = {
            "smiles": dataset.smiles[0],
            "model_version": self.model_version,
            "n_samples": self.params.conformal.n_samples,
            "coverage": COVERAGE,
            "bioactivity": bioactivity,
            "toxicity": {col: _finite(preds[col][0]) for col in TOX_COLUMNS},
        }
        check_response(response)
        return response


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(f"invalid response: {message}")


def _probability(p) -> bool:
    return p is None or (isinstance(p, float) and 0.0 <= p <= 1.0)


# same rules as the backend's parseInferenceResponse
def check_response(response: dict) -> None:
    for key in ("smiles", "model_version"):
        _check(isinstance(response[key], str) and bool(response[key]), key)
    n_samples = response["n_samples"]
    _check(isinstance(n_samples, int) and n_samples >= 2, "n_samples")
    _check(0.0 < response["coverage"] < 1.0, "coverage")
    _check(set(response["bioactivity"]) == set(BIO_TARGETS), "bioactivity targets")
    for target, bio in response["bioactivity"].items():
        value, lo, hi = bio["pic50"], bio["pic50_lo"], bio["pic50_hi"]
        if value is None:
            _check(lo is None and hi is None, f"{target} interval without pic50")
        else:
            _check(
                all(isinstance(v, float) and math.isfinite(v) for v in (value, lo, hi)),
                f"{target} pic50",
            )
            _check(lo <= value <= hi, f"{target} interval order")
            # IC50 = 10^(9 - pIC50) nM must be a finite double
            _check(9.0 - lo < 308.0, f"{target} pic50_lo")
        _check(_probability(bio["p_active"]), f"{target} p_active")
    _check(set(response["toxicity"]) == set(TOX_COLUMNS), "toxicity tasks")
    for task, p in response["toxicity"].items():
        _check(_probability(p), f"toxicity {task}")


def fetch_version(settings: Settings, seeds: Sequence[int] = SEEDS) -> None:
    # downloads only if the folder has no complete version, so restarts work offline
    directory, name, tag = (
        settings.checkpoint_dir,
        settings.model_name,
        settings.model_version,
    )
    try:
        load_tagged_version(directory, tag, name, seeds)
    except FileNotFoundError:
        download_version(settings.hf_repo, tag, directory, name, seeds)


def load_predictor(
    settings: Settings, smiles_encoder=None, seeds: Sequence[int] = SEEDS
) -> Predictor:
    if settings.hf_repo:
        fetch_version(settings, seeds)
    directory, name = settings.checkpoint_dir, settings.model_name
    # file checks first, before the slow MolFormer load
    paths = member_paths(directory, name, seeds)
    params = load_ensemble_params(paths, directory, name)
    members = load_members(
        paths, smiles_encoder or SmilesEncoder(freeze=True), device=settings.device
    )
    if uncalibrated := [m.path.name for m in members if m.calibration is None]:
        raise ValueError(f"no saved calibration: {', '.join(uncalibrated)}")
    return Predictor(
        members=members,
        params=params,
        model_version=settings.model_version
        or f"{name}_ensemble@{file_sha256(sidecar_path(directory, name))[:8]}",
    )
