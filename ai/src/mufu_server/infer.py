"""
Single-molecule MuFu inference, in the response format of `backend/docs/predict-api.md`.

pIC50 comes with its conformal MC-Dropout interval at `COVERAGE`; `p_active` and
toxicity are calibrated probabilities. MC passes are seeded, so the same SMILES always
gets the same response.
"""

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from encoders.smiles_encoder import SmilesEncoder
from mufu.calibration import CalibrationParams
from mufu.contract import ACTIVE_COLUMNS, BIO_TARGETS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import MuFuDataset
from mufu.experiment import load_mufu_checkpoint
from mufu.mc_dropout import ConformalParams, load_conformal, predict_with_intervals
from mufu.model import MuFu
from mufu.train import predict
from mufu_server.config import COVERAGE, Settings

MC_SEED = 0


class InvalidSmiles(ValueError):
    pass


@dataclass
class Predictor:
    model: MuFu
    family_states: Mapping[str, Mapping[str, torch.Tensor]] | None
    calibration: CalibrationParams | None
    conformal: ConformalParams
    model_version: str
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if COVERAGE not in self.conformal.coverages:
            raise ValueError(f"conformal params have no {COVERAGE} coverage")

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
            probs = predict(
                self.model,
                dataset,
                family_states=self.family_states,
                calibration=self.calibration,
            )
            pic50 = predict_with_intervals(self.model, dataset, self.conformal)

        pct = round(COVERAGE * 100)
        bioactivity = {}
        for target, pic50_col, active_col in zip(
            BIO_TARGETS, PIC50_COLUMNS, ACTIVE_COLUMNS
        ):
            value, lo, hi = (
                _finite(pic50[col][0])
                for col in (pic50_col, f"{pic50_col}_lo{pct}", f"{pic50_col}_hi{pct}")
            )
            if None in (value, lo, hi):
                value = lo = hi = None
            bioactivity[target] = {
                "pic50": value,
                "pic50_lo": lo,
                "pic50_hi": hi,
                "p_active": _finite(probs[active_col][0]),
            }

        response = {
            "smiles": dataset.smiles[0],
            "model_version": self.model_version,
            "n_samples": self.conformal.n_samples,
            "coverage": COVERAGE,
            "bioactivity": bioactivity,
            "toxicity": {col: _finite(probs[col][0]) for col in TOX_COLUMNS},
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


def load_predictor(settings: Settings) -> Predictor:
    path = Path(settings.checkpoint)
    model, meta = load_mufu_checkpoint(
        path, SmilesEncoder(freeze=True), device=settings.device
    )
    if meta["calibration"] is None:
        raise ValueError(f"{path.name} has no saved calibration")
    commit = meta["checkpoint"].get("git_commit")
    return Predictor(
        model=model,
        family_states=meta["family_states"],
        calibration=meta["calibration"],
        conformal=load_conformal(path),
        model_version=settings.model_version
        or (f"{path.stem}@{commit}" if commit else path.stem),
    )
