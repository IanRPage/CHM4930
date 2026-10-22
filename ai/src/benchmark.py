"""Shared benchmark foundation; importing this module never loads or writes data.

Load and split the full frozen table before selecting an endpoint. Obtain metadata
from that full table, retain it with run parameters, and verify it before exporting.
``record_results`` is the existing recorder, re-exported without schema changes;
writing results is always an explicit caller action.

Pandas prediction and mask indexes (and task columns) must match labels exactly.
NumPy arrays and Torch tensors use positional alignment. Missing labels are masked
automatically; filled MuFu labels still require their contract's explicit masks.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.results import RESULTS_PATH, file_sha256, record_results
from evaluation import evaluation_metrics
from pipeline.bioactivity_loader import PIC50_ACTIVE_THRESHOLD
from pipeline.combined_loader import (
    COMBINED_CSV_PATH,
    TARGETS,
    TOX_TASKS,
    load_combined,
)
from pipeline.split import SPLITS, scaffold_split

BENCHMARK_VERSION = 1
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SEED = None
TRAINING_SEED = 42
ACTIVE_THRESHOLD = PIC50_ACTIVE_THRESHOLD

__all__ = [
    "benchmark_metadata",
    "config_json",
    "evaluate_predictions",
    "load_benchmark",
    "record_results",
    "set_training_seed",
    "verify_benchmark_metadata",
    "verify_export",
]


def config_json(config) -> str:
    return json.dumps(config, sort_keys=True)


def load_benchmark(data_path: Path = COMBINED_CSV_PATH) -> pd.DataFrame:
    """Read the frozen snapshot and split every row with the existing RF policy."""
    return scaffold_split(
        load_combined(path=data_path, threshold=ACTIVE_THRESHOLD),
        frac=SPLIT_FRACTIONS,
        seed=SPLIT_SEED,
    )


def benchmark_metadata(df: pd.DataFrame, data_path: Path = COMBINED_CSV_PATH) -> dict:
    """Identify the full dataset and its assignments, independent of row order.

    Reject filtered or modified labels/source flags: metadata belongs to the full
    snapshot. The split hash serializes sorted (SMILES, split) pairs as UTF-8 JSON,
    independently of DataFrame indexes and platform line endings. Training seed
    describes the canonical protocol; exploratory run seeds belong in run params.
    """
    snapshot = load_combined(path=data_path, threshold=ACTIVE_THRESHOLD)
    required = {*snapshot.columns, "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"benchmark table is missing columns: {sorted(missing)}")
    if df["smiles"].isna().any() or df["smiles"].duplicated().any():
        raise ValueError("benchmark requires unique, nonmissing SMILES")
    if not df["split"].isin(SPLITS).all():
        raise ValueError("benchmark split must be train, val, or test")

    ordered = df.sort_values("smiles").reset_index(drop=True)
    snapshot = snapshot.sort_values("smiles").reset_index(drop=True)
    if not ordered[snapshot.columns].equals(snapshot):
        raise ValueError("benchmark metadata requires the full, unchanged snapshot")
    assignments = ordered[["smiles", "split"]].to_numpy().tolist()
    split_bytes = json.dumps(
        assignments, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "data_sha256": file_sha256(data_path),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "active_threshold": ACTIVE_THRESHOLD,
        "n_molecules": len(df),
        "bio_targets": list(TARGETS),
        "tox_tasks": list(TOX_TASKS),
        "metrics": {
            "regression": ["rmse", "r2"],
            "classification": ["roc_auc"],
            "multitask": ["per_task_roc_auc", "mean_roc_auc"],
        },
    }


def verify_benchmark_metadata(
    expected: dict, df: pd.DataFrame, data_path: Path = COMBINED_CSV_PATH
) -> None:
    """Raise if stored metadata differs from the current full benchmark table."""
    actual = benchmark_metadata(df, data_path)
    differences = sorted(
        key
        for key in expected.keys() | actual.keys()
        if key not in expected or key not in actual or expected[key] != actual[key]
    )
    if differences:
        raise ValueError(f"benchmark metadata mismatch: {', '.join(differences)}")


def verify_export(
    metadata: dict,
    df: pd.DataFrame,
    data_path: Path = COMBINED_CSV_PATH,
    results_path: Path = RESULTS_PATH,
) -> None:
    """Check provenance and refuse an export that would discard existing RF rows."""
    verify_benchmark_metadata(metadata, df, data_path)
    if results_path.exists():
        existing = pd.read_csv(results_path)
        rf = existing[existing["model"].isin(["rf_bioactivity", "rf_toxicity"])]
        if not rf["data_sha256"].eq(metadata["data_sha256"]).all():
            raise ValueError("export would discard RF results from another snapshot")


def set_training_seed(seed: int = TRAINING_SEED) -> torch.Generator:
    """Seed Python/NumPy/Torch and return a seeded CPU DataLoader generator.

    RF callers also pass this seed as random_state. This does not change Torch's
    deterministic-algorithm settings or promise reproducibility across hardware.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("training seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _check_alignment(labels, values, name: str) -> None:
    if isinstance(labels, (pd.Series, pd.DataFrame)) and isinstance(
        values, (pd.Series, pd.DataFrame)
    ):
        if not labels.index.equals(values.index):
            raise ValueError(f"{name} row index must match labels")
        if (
            isinstance(labels, pd.DataFrame)
            and isinstance(values, pd.DataFrame)
            and not labels.columns.equals(values.columns)
        ):
            raise ValueError(f"{name} task columns must match labels")


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def evaluate_predictions(y_true, y_score, task_type: str, mask=None) -> dict:
    """Evaluate aligned continuous scores with the shared metrics and label masks.

    Supplied masks must contain only booleans and are intersected with observed
    labels. Nonfinite observed labels or predictions are errors, rather than
    silently reducing the evaluation cohort.
    Use predicted pIC50 for the existing bioactivity ROC-AUC and continuous positive
    class scores for toxicity. Undefined metrics retain evaluation_metrics' NaNs.
    """
    _check_alignment(y_true, y_score, "prediction")
    if mask is not None:
        _check_alignment(y_true, mask, "mask")
    labels, scores = _as_numpy(y_true), _as_numpy(y_score)
    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")
    observed = ~np.isnan(labels)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                raise ValueError("mask must contain only boolean values")
            mask = mask.detach().cpu().numpy()
        supplied = np.asarray(mask)
        if supplied.shape != labels.shape:
            raise ValueError("mask must have the same shape as y_true")
        if supplied.dtype != bool and not all(
            isinstance(value, (bool, np.bool_)) for value in supplied.flat
        ):
            raise ValueError("mask must contain only boolean values")
        observed &= supplied.astype(bool)
    if not np.isfinite(labels[observed]).all():
        raise ValueError("observed labels must be finite")
    if not np.isfinite(scores[observed]).all():
        raise ValueError("observed predictions must be finite")
    return evaluation_metrics(labels, scores, task_type, mask=observed)
