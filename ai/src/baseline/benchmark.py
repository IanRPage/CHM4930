"""
Shared benchmark setup so every model is evaluated on the same split, rows, and seeds.

Models provide a `fit_predict(df, seed)` callable that returns a mapping of split name
(`"val"`, `"test"`) to `{label_column: scores}` aligned with `df[df["split"] == split]`.
`run_benchmark()` runs `fit_predict` across `SEEDS`, evaluates predictions with the
dataset label masks, records each seed's rows to `ai/data/baseline/results.csv` as soon as
it finishes (so a crash keeps the seeds already done), and returns the seed-aggregated
summary table.
"""

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.results import (
    RESULTS_PATH,
    params_json,
    record_results,
    summarize_seeds,
)
from evaluation import evaluation_metrics
from pipeline.combined_loader import (
    COMBINED_CSV_PATH,
    TARGETS,
    TOX_TASKS,
    load_combined,
)
from pipeline.split import SPLITS, scaffold_split
from pipeline.toxicity_loader import DATASETS

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

BIO_PIC50_COLS = tuple(f"pIC50_{t}" for t in TARGETS)
BIO_ACTIVE_COLS = tuple(f"active_{t}" for t in TARGETS)
VALID_PREDICTION_COLS = frozenset([*BIO_PIC50_COLS, *BIO_ACTIVE_COLS, *TOX_TASKS])
TOX_DATASET_BY_TASK = {
    task: dataset for dataset, config in DATASETS.items() for task in config["tasks"]
}


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def load_benchmark_split(data_path: Path = COMBINED_CSV_PATH) -> pd.DataFrame:
    return scaffold_split(load_combined(path=data_path))


def split_sha256(df: pd.DataFrame) -> str:
    pairs = df.sort_values("smiles")[["smiles", "split"]].to_numpy().tolist()
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def _validated_scores(
    scores,
    expected_len: int,
    labeled_mask: np.ndarray,
    split: str,
    column: str,
) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.shape != (expected_len,):
        raise ValueError(
            f"{split}/{column}: expected 1D array of length {expected_len}, "
            f"got shape {arr.shape}"
        )
    if not np.isfinite(arr[labeled_mask]).all():
        raise ValueError(f"{split}/{column}: non-finite score on labeled row(s)")
    return arr


def evaluate_split_predictions(
    df: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
) -> list[dict]:
    rows: list[dict] = []

    for split, split_preds in predictions.items():
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        unknown = set(split_preds) - VALID_PREDICTION_COLS - {"smiles"}
        if unknown:
            raise ValueError(f"{split}: unknown prediction column(s) {sorted(unknown)}")

        split_df = df[df["split"] == split]
        if (
            "smiles" in split_preds
            and list(split_preds["smiles"]) != split_df["smiles"].tolist()
        ):
            raise ValueError(f"{split}: prediction SMILES do not match split SMILES")
        n_rows = len(split_df)

        for target in TARGETS:
            pic50_col = f"pIC50_{target}"
            active_col = f"active_{target}"
            if pic50_col not in split_preds and active_col not in split_preds:
                continue

            mask = split_df[pic50_col].notna().to_numpy()
            y_reg = split_df.loc[mask, pic50_col].to_numpy(dtype=float)
            y_cls = split_df.loc[mask, active_col].to_numpy(dtype=float)
            n_labeled = int(mask.sum())
            n_positive = int(np.sum(y_cls == 1))

            if pic50_col in split_preds:
                pic50_scores = _validated_scores(
                    split_preds[pic50_col], n_rows, mask, split, pic50_col
                )
                reg = evaluation_metrics(
                    y_reg, pic50_scores[mask], task_type="regression"
                )
                cls = evaluation_metrics(
                    y_cls, pic50_scores[mask], task_type="classification"
                )
                for metric, value in (
                    ("rmse", reg["rmse"]),
                    ("r2", reg["r2"]),
                    ("roc_auc", cls["roc_auc"]),
                ):
                    rows.append(
                        {
                            "dataset": target.lower(),
                            "endpoint": pic50_col,
                            "split": split,
                            "metric": metric,
                            "value": value,
                            "n_labeled": n_labeled,
                            "n_positive": n_positive,
                        }
                    )

            if active_col in split_preds:
                active_scores = _validated_scores(
                    split_preds[active_col], n_rows, mask, split, active_col
                )
                cls = evaluation_metrics(
                    y_cls, active_scores[mask], task_type="classification"
                )
                rows.append(
                    {
                        "dataset": target.lower(),
                        "endpoint": active_col,
                        "split": split,
                        "metric": "roc_auc",
                        "value": cls["roc_auc"],
                        "n_labeled": n_labeled,
                        "n_positive": n_positive,
                    }
                )

        predicted_tox = [task for task in TOX_TASKS if task in split_preds]
        if predicted_tox:
            y_tox = split_df[TOX_TASKS].to_numpy(dtype=float)
            observed = split_df[TOX_TASKS].notna().to_numpy()
            scores_matrix = np.zeros((n_rows, len(TOX_TASKS)), dtype=float)
            trained = np.zeros(len(TOX_TASKS), dtype=bool)

            for j, task in enumerate(TOX_TASKS):
                if task not in split_preds:
                    continue
                obs = observed[:, j]
                task_scores = _validated_scores(
                    split_preds[task], n_rows, obs, split, task
                )
                scores_matrix[obs, j] = task_scores[obs]
                trained[j] = True

            mask = observed & trained
            overall = evaluation_metrics(
                y_tox, scores_matrix, task_type="multitask", mask=mask
            )
            groups = {
                "overall": list(range(len(TOX_TASKS))),
                **{
                    dataset: [TOX_TASKS.index(t) for t in config["tasks"]]
                    for dataset, config in DATASETS.items()
                },
            }
            for group, cols in groups.items():
                result = (
                    overall
                    if group == "overall"
                    else evaluation_metrics(
                        y_tox[:, cols],
                        scores_matrix[:, cols],
                        task_type="multitask",
                        mask=mask[:, cols],
                    )
                )
                rows.append(
                    {
                        "dataset": group,
                        "endpoint": f"macro_{group}",
                        "split": split,
                        "metric": "roc_auc",
                        "value": result["mean_roc_auc"],
                        "n_labeled": np.nan,
                        "n_positive": np.nan,
                    }
                )

            for j, task in enumerate(TOX_TASKS):
                if not trained[j]:
                    continue
                obs = observed[:, j]
                rows.append(
                    {
                        "dataset": TOX_DATASET_BY_TASK[task],
                        "endpoint": task,
                        "split": split,
                        "metric": "roc_auc",
                        "value": overall["per_task_roc_auc"][j],
                        "n_labeled": int(obs.sum()),
                        "n_positive": int(np.sum(y_tox[obs, j] == 1)),
                    }
                )

    return rows


def run_benchmark(
    fit_predict: Callable[
        [pd.DataFrame, int],
        Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    ],
    model: str,
    encoder: str,
    params: dict,
    *,
    seeds: Sequence[int] | None = None,
    df: pd.DataFrame | None = None,
    path: Path = RESULTS_PATH,
    data_path: Path = COMBINED_CSV_PATH,
    allow_drop_stale: bool = False,
) -> pd.DataFrame:
    if seeds is None:
        seeds = SEEDS
    if not seeds:
        raise ValueError("seeds must not be empty")
    if df is None:
        df = load_benchmark_split(data_path)

    encoded_params = params_json(params)
    for seed in seeds:
        seed_everything(seed)
        predictions = fit_predict(df, seed)
        missing = [s for s in ("val", "test") if s not in predictions]
        if missing:
            raise ValueError(
                "fit_predict must return 'val' and 'test' predictions; "
                f"missing {missing}"
            )
        eval_preds = {split: predictions[split] for split in ("val", "test")}
        rows = [
            {**row, "seed": int(seed), "encoder": encoder, "params": encoded_params}
            for row in evaluate_split_predictions(df, eval_preds)
        ]
        recorded = record_results(
            rows,
            model=model,
            path=path,
            data_path=data_path,
            allow_drop_stale=allow_drop_stale,
        )

    return summarize_seeds(
        recorded[(recorded["model"] == model) & (recorded["encoder"] == encoder)]
    )
