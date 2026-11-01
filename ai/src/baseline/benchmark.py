"""
Shared benchmark setup so every model is evaluated on the same split, rows, and seeds.

Models provide a `fit_predict(df, seed)` callable that returns a mapping of split name
(`"val"`, `"test"`) to `{label_column: scores}` aligned with `df[df["split"] == split]`.
`run_benchmark()` runs `fit_predict` across `SEEDS` on one scaffold split, evaluates
predictions with the dataset label masks and a bootstrap CI over test compounds, records
each seed's rows to `ai/data/baseline/results.csv` and its test predictions to
`ai/outputs/predictions/` as it finishes, and returns the seed-aggregated summary table.

Split ids: `"default"` sends the largest scaffold groups to train, `"scaffold-<k>"` is
the balanced split shuffled with seed `k`.
"""

import hashlib
import json
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.results import (
    DEFAULT_SPLIT_ID,
    RESULTS_PATH,
    git_commit,
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
SPLIT_IDS: tuple[str, ...] = (
    DEFAULT_SPLIT_ID,
    *(f"scaffold-{k}" for k in range(5)),
)
N_BOOT = 2000
PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "predictions"

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


def load_benchmark_split(
    data_path: Path = COMBINED_CSV_PATH, split_id: str = DEFAULT_SPLIT_ID
) -> pd.DataFrame:
    combined = load_combined(path=data_path)
    if split_id == DEFAULT_SPLIT_ID:
        return scaffold_split(combined)
    prefix, _, k = split_id.partition("-")
    if prefix != "scaffold" or not k.isdigit():
        raise ValueError(
            f"split_id must be 'default' or 'scaffold-<k>', got {split_id!r}"
        )
    return scaffold_split(combined, seed=int(k))


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


def _split_tables(split_df: pd.DataFrame, split_preds: Mapping, split: str):
    """
    Validated labels and scores per predicted endpoint, row-aligned with `split_df`.
    """
    n_rows = len(split_df)
    bio = []
    for target in TARGETS:
        pic50_col = f"pIC50_{target}"
        active_col = f"active_{target}"
        if pic50_col not in split_preds and active_col not in split_preds:
            continue
        labeled = split_df[pic50_col].notna().to_numpy()
        entry = {
            "target": target,
            "labeled": labeled,
            "y_reg": split_df[pic50_col].to_numpy(dtype=float),
            "y_cls": split_df[active_col].to_numpy(dtype=float),
            "scores": {},
        }
        for col in (pic50_col, active_col):
            if col in split_preds:
                entry["scores"][col] = _validated_scores(
                    split_preds[col], n_rows, labeled, split, col
                )
        bio.append(entry)

    tox = None
    if any(task in split_preds for task in TOX_TASKS):
        observed = split_df[TOX_TASKS].notna().to_numpy()
        scores = np.zeros((n_rows, len(TOX_TASKS)), dtype=float)
        trained = np.zeros(len(TOX_TASKS), dtype=bool)
        for j, task in enumerate(TOX_TASKS):
            if task not in split_preds:
                continue
            obs = observed[:, j]
            scores[obs, j] = _validated_scores(
                split_preds[task], n_rows, obs, split, task
            )[obs]
            trained[j] = True
        tox = {
            "y": split_df[TOX_TASKS].to_numpy(dtype=float),
            "mask": observed & trained,
            "scores": scores,
            "trained": trained,
        }
    return bio, tox


def _score(bio: list[dict], tox: dict | None, sel: np.ndarray) -> list[dict]:
    """
    Metric rows over split rows `sel`, in an order that depends only on which endpoints
    were predicted, so bootstrap resamples line up row for row.
    """
    rows: list[dict] = []

    def add(dataset, endpoint, metric, value, **extra):
        row = {"dataset": dataset, "endpoint": endpoint, "metric": metric}
        rows.append({**row, "value": value, **extra})

    for entry in bio:
        target = entry["target"]
        labeled = entry["labeled"][sel]
        y_reg = entry["y_reg"][sel][labeled]
        y_cls = entry["y_cls"][sel][labeled]
        n_labeled = int(labeled.sum())
        n_positive = int(np.sum(y_cls == 1))
        counts = {"n_labeled": n_labeled, "n_positive": n_positive}
        prevalence = n_positive / n_labeled if n_labeled else float("nan")
        for col, col_scores in entry["scores"].items():
            scores = col_scores[sel][labeled]
            metrics = {}
            if col.startswith("pIC50_"):
                metrics |= evaluation_metrics(y_reg, scores, task_type="regression")
            metrics |= evaluation_metrics(y_cls, scores, task_type="classification")
            metrics["prevalence"] = prevalence
            for metric, value in metrics.items():
                add(target.lower(), col, metric, value, **counts)

    if tox is not None:
        y, mask, trained = tox["y"][sel], tox["mask"][sel], tox["trained"]
        overall = evaluation_metrics(
            y, tox["scores"][sel], task_type="multitask", mask=mask
        )
        roc = np.asarray(overall["per_task_roc_auc"])
        pr = np.asarray(overall["per_task_pr_auc"])
        groups = {
            "overall": list(range(len(TOX_TASKS))),
            **{
                dataset: [TOX_TASKS.index(t) for t in config["tasks"]]
                for dataset, config in DATASETS.items()
            },
        }
        for group, cols in groups.items():
            kept = [j for j in cols if not np.isnan(roc[j])]
            dropped = ";".join(TOX_TASKS[j] for j in cols if j not in kept) or np.nan
            extra = {
                "n_labeled": np.nan,
                "n_positive": np.nan,
                "n_tasks": len(kept),
                "dropped_tasks": dropped,
            }
            for metric, per_task in (("roc_auc", roc), ("pr_auc", pr)):
                value = float(np.mean(per_task[kept])) if kept else float("nan")
                add(group, f"macro_{group}", metric, value, **extra)

        for j, task in enumerate(TOX_TASKS):
            if not trained[j]:
                continue
            obs = mask[:, j]
            n_labeled = int(obs.sum())
            n_positive = int(np.sum(y[obs, j] == 1))
            counts = {"n_labeled": n_labeled, "n_positive": n_positive}
            dataset = TOX_DATASET_BY_TASK[task]
            add(dataset, task, "roc_auc", roc[j], **counts)
            add(dataset, task, "pr_auc", pr[j], **counts)
            prevalence = n_positive / n_labeled if n_labeled else float("nan")
            add(dataset, task, "prevalence", prevalence, **counts)

    return rows


def _bootstrap_ci(
    bio: list[dict], tox: dict | None, n_rows: int, n_boot: int, seed: int
) -> np.ndarray:
    """
    2.5/97.5 percentiles of each `_score` row over `n_boot` compound resamples, shared
    by every endpoint. Draws where a metric is undefined are skipped.
    """
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            [row["value"] for row in _score(bio, tox, rng.integers(0, n_rows, n_rows))]
            for _ in range(n_boot)
        ],
        dtype=float,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(draws, [2.5, 97.5], axis=0)


def evaluate_split_predictions(
    df: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    *,
    n_boot: int = 0,
    boot_seed: int = 0,
) -> list[dict]:
    """
    With `n_boot > 0`, test rows get a bootstrap CI over test compounds in
    `ci_low`/`ci_high`; every other row gets NaN.
    """
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

        bio, tox = _split_tables(split_df, split_preds, split)
        split_rows = _score(bio, tox, np.arange(n_rows))
        if n_boot > 0 and split == "test" and n_rows and split_rows:
            ci_low, ci_high = _bootstrap_ci(bio, tox, n_rows, n_boot, boot_seed)
        else:
            ci_low = ci_high = np.full(len(split_rows), np.nan)
        rows += [
            {**row, "split": split, "ci_low": float(low), "ci_high": float(high)}
            for row, low, high in zip(split_rows, ci_low, ci_high)
        ]

    return rows


def prediction_path(
    model: str,
    encoder: str,
    split_id: str,
    seed: int,
    predictions_dir: Path | None = None,
) -> Path:
    directory = PREDICTIONS_DIR if predictions_dir is None else predictions_dir
    return directory / f"{model}__{encoder}__{split_id}__seed{seed}.csv"


def save_predictions(
    df: pd.DataFrame, test_preds: Mapping[str, Sequence[float]], path: Path
) -> None:
    out = pd.DataFrame({"smiles": df.loc[df["split"] == "test", "smiles"].to_numpy()})
    for col, scores in test_preds.items():
        if col != "smiles":
            out[col] = np.asarray(scores, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


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
    split_id: str = DEFAULT_SPLIT_ID,
    df: pd.DataFrame | None = None,
    path: Path = RESULTS_PATH,
    data_path: Path = COMBINED_CSV_PATH,
    allow_drop_stale: bool = False,
    n_boot: int | None = None,
    predictions_dir: Path | None = None,
    commit: str | None = None,
) -> pd.DataFrame:
    """
    `df`, if given, must already be split as `split_id`. `commit` is the commit that
    trained the model, for scoring saved checkpoints; the current one by default.
    """
    if seeds is None:
        seeds = SEEDS
    if not seeds:
        raise ValueError("seeds must not be empty")
    if n_boot is None:
        n_boot = N_BOOT
    if df is None:
        df = load_benchmark_split(data_path, split_id)

    encoded_params = params_json(params)
    split_hash = split_sha256(df)
    commit = git_commit() if commit is None else commit
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
            {
                **row,
                "split_id": split_id,
                "seed": int(seed),
                "encoder": encoder,
                "params": encoded_params,
                "split_sha256": split_hash,
            }
            for row in evaluate_split_predictions(
                df, eval_preds, n_boot=n_boot, boot_seed=int(seed)
            )
        ]
        recorded = record_results(
            rows,
            model=model,
            path=path,
            data_path=data_path,
            allow_drop_stale=allow_drop_stale,
            commit=commit,
        )
        save_predictions(
            df,
            eval_preds["test"],
            prediction_path(model, encoder, split_id, int(seed), predictions_dir),
        )

    return summarize_seeds(
        recorded[
            (recorded["model"] == model)
            & (recorded["encoder"] == encoder)
            & (recorded["split_id"] == split_id)
        ]
    )
