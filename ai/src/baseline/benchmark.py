"""
Shared benchmark setup so every model is evaluated on the same split, rows, and seeds.

Models provide `fit_predict(df, seed)` returning `"val"` and `"test"` predictions
(`{column: scores}`) aligned with `df[df["split"] == split]`. `run_benchmark()` runs
`fit_predict` across `SEEDS`, saves predictions to `ai/runs/<run_id>/predictions/`,
scores `val` only, and returns the validation seed summary.

`promote_run()` and `rebuild_tracked_results()` score saved `test` predictions and
write `ai/data/baseline/*.csv`.
"""

import datetime
import hashlib
import json
import logging
import multiprocessing as mp
import random
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.results import (
    COLUMNS,
    DEFAULT_SPLIT_ID,
    METRIC_COLUMNS,
    RESULTS_PATH,
    SORT_KEYS,
    CodeIdentity,
    code_identity,
    file_sha256,
    params_json,
    summarize_seeds,
    write_results,
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

log = logging.getLogger(__name__)

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SPLIT_IDS: tuple[str, ...] = (
    DEFAULT_SPLIT_ID,
    *(f"scaffold-{k}" for k in range(5)),
)
N_BOOT = 2000
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
PROMOTED_PATH = COMBINED_CSV_PATH.with_name("promoted.json")

BIO_PIC50_COLS = tuple(f"pIC50_{t}" for t in TARGETS)
BIO_ACTIVE_COLS = tuple(f"active_{t}" for t in TARGETS)
VALID_PREDICTION_COLS = frozenset([*BIO_PIC50_COLS, *BIO_ACTIVE_COLS, *TOX_TASKS])
TOX_DATASET_BY_TASK = {
    task: dataset for dataset, config in DATASETS.items() for task in config["tasks"]
}
_INTERVAL_COL_RE = re.compile(r"^(.+)_(sigma|lo\d+|hi\d+)$")


def _is_auxiliary_interval_col(col: str) -> bool:
    m = _INTERVAL_COL_RE.match(col)
    return bool(m and m.group(1) in VALID_PREDICTION_COLS)


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


def make_run_id(
    model: str, split_id: str = DEFAULT_SPLIT_ID, *, run_name: str = ""
) -> str:
    if run_name:
        return f"{model}_{run_name}_{split_id}"
    if model.startswith("rf_"):
        return f"{model}_{split_id}"
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return f"{model}_{stamp}_{split_id}"


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


_FORK_BIO: list[dict] = []
_FORK_TOX: dict | None = None


def _score_boot_chunk(chunk: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [[row["value"] for row in _score(_FORK_BIO, _FORK_TOX, sel)] for sel in chunk],
        dtype=float,
    )


def _fork_boot_chunks(
    chunk_fn: Callable[[list[np.ndarray]], np.ndarray],
    sels: list[np.ndarray],
) -> np.ndarray:
    if len(sels) >= 200 and "fork" in mp.get_all_start_methods():
        chunk_size = max(1, (len(sels) + 19) // 20)
        chunks = [sels[i : i + chunk_size] for i in range(0, len(sels), chunk_size)]
        with mp.get_context("fork").Pool() as pool:
            return np.vstack(pool.map(chunk_fn, chunks))
    return chunk_fn(sels)


def _bootstrap_ci(
    bio: list[dict], tox: dict | None, n_rows: int, n_boot: int, seed: int
) -> np.ndarray:
    """
    2.5/97.5 percentiles of each `_score` row over `n_boot` compound resamples, shared
    by every endpoint. Draws where a metric is undefined are skipped.
    """
    global _FORK_BIO, _FORK_TOX
    _FORK_BIO, _FORK_TOX = bio, tox
    rng = np.random.default_rng(seed)
    sels = [rng.integers(0, n_rows, n_rows) for _ in range(n_boot)]
    draws = _fork_boot_chunks(_score_boot_chunk, sels)
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
        unknown = {
            c
            for c in split_preds
            if c not in VALID_PREDICTION_COLS
            and c != "smiles"
            and not _is_auxiliary_interval_col(c)
        }
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


def verify_predictions_df_match(new_df: pd.DataFrame, existing_path: Path) -> None:
    old_df = pd.read_csv(existing_path, float_precision="round_trip")
    if list(old_df.columns) != list(new_df.columns):
        raise FileExistsError(
            f"refusing to overwrite write-once {existing_path}: column mismatch "
            f"({list(old_df.columns)} vs {list(new_df.columns)})"
        )
    if old_df["smiles"].tolist() != new_df["smiles"].tolist():
        raise FileExistsError(
            f"refusing to overwrite write-once {existing_path}: SMILES mismatch"
        )
    for col in new_df.columns:
        if col == "smiles":
            continue
        a = old_df[col].to_numpy(dtype=float)
        b = new_df[col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True):
            max_diff = float(np.nanmax(np.abs(a - b)))
            raise FileExistsError(
                f"refusing to overwrite write-once {existing_path}: column {col!r} "
                f"differs (max abs diff {max_diff:.3e})"
            )


def save_split_predictions(
    df: pd.DataFrame,
    split: str,
    split_preds: Mapping[str, Sequence[float] | np.ndarray],
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    out = pd.DataFrame({"smiles": df.loc[df["split"] == split, "smiles"].to_numpy()})
    for col, scores in split_preds.items():
        if col != "smiles":
            out[col] = np.asarray(scores, dtype=float)
    if path.exists() and not overwrite:
        verify_predictions_df_match(out, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")


def load_split_predictions(path: Path) -> dict[str, np.ndarray | list[str]]:
    df_pred = pd.read_csv(path, float_precision="round_trip")
    return {
        "smiles": df_pred["smiles"].tolist(),
        **{
            c: df_pred[c].to_numpy(dtype=float)
            for c in df_pred.columns
            if c in VALID_PREDICTION_COLS
        },
    }


def _write_run_json(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(run_dir / "run.json")


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
    data_path: Path = COMBINED_CSV_PATH,
    run_dir: Path | None = None,
    runs_dir: Path | None = None,
    run_name: str = "",
    identity: CodeIdentity | None = None,
    mark_complete: bool = True,
    overwrite_predictions: bool = False,
    promote: bool = False,
) -> pd.DataFrame:
    """Run `fit_predict` across `seeds`, save predictions to `run_dir`, and return `val` summary."""
    if seeds is None:
        seeds = SEEDS
    if not seeds:
        raise ValueError("seeds must not be empty")
    if df is None:
        df = load_benchmark_split(data_path, split_id)
    if identity is None:
        identity = code_identity()
    if runs_dir is None:
        runs_dir = RUNS_DIR

    if run_dir is None:
        run_id = make_run_id(model, split_id, run_name=run_name)
        run_dir = runs_dir / run_id
    else:
        run_id = run_dir.name

    encoded_params = params_json(params)
    data_sha = file_sha256(data_path)
    split_hash = split_sha256(df)
    date_str = datetime.datetime.now(datetime.UTC).date().isoformat()
    existing_meta_path = run_dir / "run.json"
    if existing_meta_path.exists():
        old_meta = json.loads(existing_meta_path.read_text())
        if "date" in old_meta:
            date_str = str(old_meta["date"])

    run_meta = {
        "run_id": run_id,
        "model": model,
        "encoder": encoder,
        "split_id": split_id,
        "seeds": [int(s) for s in seeds],
        "completed_seeds": [],
        "status": "running",
        "params": encoded_params,
        "data_sha256": data_sha,
        "split_sha256": split_hash,
        "base_commit": identity.base_commit,
        "ai_tree": identity.ai_tree,
        "patch_sha256": identity.patch_sha256,
        "date": date_str,
    }

    val_rows: list[dict] = []
    meta_initialized = False

    for seed in seeds:
        seed_everything(int(seed))
        predictions = fit_predict(df, int(seed))
        missing = [s for s in ("val", "test") if s not in predictions]
        if missing:
            raise ValueError(
                "fit_predict must return 'val' and 'test' predictions; "
                f"missing {missing}"
            )
        seed_val_rows = [
            {
                **row,
                "model": model,
                "encoder": encoder,
                "split_id": split_id,
                "seed": int(seed),
            }
            for row in evaluate_split_predictions(
                df, {"val": predictions["val"]}, n_boot=0, boot_seed=int(seed)
            )
        ]
        evaluate_split_predictions(
            df, {"test": predictions["test"]}, n_boot=0, boot_seed=int(seed)
        )

        if not meta_initialized:
            _write_run_json(run_dir, run_meta)
            if identity.patch:
                patch_path = run_dir / "patch.diff"
                patch_bytes = identity.patch.encode("utf-8")
                if (
                    patch_path.exists()
                    and not overwrite_predictions
                    and patch_path.read_bytes() != patch_bytes
                ):
                    raise FileExistsError(f"refusing to overwrite {patch_path}")
                patch_path.write_bytes(patch_bytes)
            meta_initialized = True

        preds_dir = run_dir / "predictions"
        save_split_predictions(
            df,
            "val",
            predictions["val"],
            preds_dir / f"val_seed_{int(seed)}.csv",
            overwrite=overwrite_predictions,
        )
        save_split_predictions(
            df,
            "test",
            predictions["test"],
            preds_dir / f"test_seed_{int(seed)}.csv",
            overwrite=overwrite_predictions,
        )

        val_rows.extend(seed_val_rows)
        val_df = (
            pd.DataFrame(val_rows)
            .reindex(columns=METRIC_COLUMNS)
            .sort_values(SORT_KEYS, ignore_index=True)
        )
        val_df["seed"] = val_df["seed"].astype("Int64")
        val_df.to_csv(run_dir / "val_metrics.csv", index=False, lineterminator="\n")

        run_meta["completed_seeds"].append(int(seed))
        _write_run_json(run_dir, run_meta)

    if mark_complete:
        run_meta["status"] = "complete"
        _write_run_json(run_dir, run_meta)

    if promote:
        scored = score_run_predictions(
            run_dir, df, include_test=True, n_boot=0, data_path=data_path
        )
        promote_run(run_dir, runs_dir=runs_dir, data_path=data_path, rebuild=False)
        return summarize_seeds(scored)

    full_val_df = val_df.assign(
        run_id=run_id,
        n_seeds=len(seeds),
        params=encoded_params,
        data_sha256=data_sha,
        split_sha256=split_hash,
        base_commit=identity.base_commit,
        ai_tree=identity.ai_tree,
        patch_sha256=identity.patch_sha256,
        date=date_str,
    ).reindex(columns=COLUMNS)
    return summarize_seeds(full_val_df)


def score_run_predictions(
    run_dir: Path,
    df: pd.DataFrame | None = None,
    *,
    include_test: bool = False,
    n_boot: int = N_BOOT,
    data_path: Path = COMBINED_CSV_PATH,
) -> pd.DataFrame:
    """Score saved `val` (`n_boot=0`) and optional `test` (`n_boot=n_boot`) predictions in `run_dir`."""
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing run.json in {run_dir}")
    meta = json.loads(meta_path.read_text())
    if include_test and meta.get("status") != "complete":
        raise ValueError(
            f"refusing to score test on incomplete run {run_dir.name} "
            f"(status={meta.get('status')!r})"
        )
    seeds = [int(s) for s in meta.get("seeds", SEEDS)]
    if include_test and list(meta.get("completed_seeds", seeds)) != seeds:
        raise ValueError(
            f"refusing to score test on run {run_dir.name}: completed_seeds "
            f"{meta.get('completed_seeds')} != {seeds}"
        )

    split_id = str(meta["split_id"])
    if df is None:
        df = load_benchmark_split(data_path, split_id)

    model = str(meta["model"])
    encoder = str(meta["encoder"])
    preds_dir = run_dir / "predictions"

    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for seed in seeds:
        val_path = preds_dir / f"val_seed_{seed}.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"missing validation predictions {val_path}")
        val_preds = load_split_predictions(val_path)
        val_rows += [
            {
                **row,
                "model": model,
                "encoder": encoder,
                "split_id": split_id,
                "seed": seed,
            }
            for row in evaluate_split_predictions(
                df, {"val": val_preds}, n_boot=0, boot_seed=seed
            )
        ]
        if include_test:
            test_path = preds_dir / f"test_seed_{seed}.csv"
            if not test_path.exists():
                raise FileNotFoundError(f"missing test predictions {test_path}")
            test_preds = load_split_predictions(test_path)
            test_rows += [
                {
                    **row,
                    "model": model,
                    "encoder": encoder,
                    "split_id": split_id,
                    "seed": seed,
                }
                for row in evaluate_split_predictions(
                    df, {"test": test_preds}, n_boot=n_boot, boot_seed=seed
                )
            ]

    val_df = (
        pd.DataFrame(val_rows)
        .reindex(columns=METRIC_COLUMNS)
        .sort_values(SORT_KEYS, ignore_index=True)
    )
    val_df["seed"] = val_df["seed"].astype("Int64")
    val_df.to_csv(run_dir / "val_metrics.csv", index=False, lineterminator="\n")

    if include_test:
        test_df = (
            pd.DataFrame(test_rows)
            .reindex(columns=METRIC_COLUMNS)
            .sort_values(SORT_KEYS, ignore_index=True)
        )
        test_df["seed"] = test_df["seed"].astype("Int64")
        test_df.to_csv(run_dir / "test_metrics.csv", index=False, lineterminator="\n")
        combined_metrics = pd.concat([val_df, test_df], ignore_index=True)
    else:
        combined_metrics = val_df

    out = (
        combined_metrics.assign(
            run_id=str(meta["run_id"]),
            n_seeds=len(seeds),
            params=str(meta["params"]),
            data_sha256=str(meta["data_sha256"]),
            split_sha256=str(meta["split_sha256"]),
            base_commit=str(meta["base_commit"]),
            ai_tree=str(meta["ai_tree"]),
            patch_sha256=str(meta.get("patch_sha256", "")),
            date=str(meta["date"]),
        )
        .reindex(columns=COLUMNS)
        .sort_values(SORT_KEYS, ignore_index=True)
    )
    out["seed"] = out["seed"].astype("Int64")
    out["n_seeds"] = out["n_seeds"].astype("Int64")
    return out


def resolve_run_dir(
    run_id: str = "",
    *,
    model: str = "mufu",
    split_id: str = DEFAULT_SPLIT_ID,
    promoted_path: Path | None = None,
    runs_dir: Path | None = None,
) -> Path:
    """Resolve `run_id` under `runs_dir`, defaulting to the promoted `(model, split_id)` run."""
    base = runs_dir or RUNS_DIR
    if run_id:
        return base / run_id
    prom = promoted_path or PROMOTED_PATH
    if prom.exists():
        promoted = json.loads(prom.read_text())
        default_id = promoted.get(model, {}).get(split_id)
        if default_id:
            return base / str(default_id)
    raise FileNotFoundError(
        "Set `run_id` at the top of the file or promote a default MuFu run first."
    )


def is_promoted_run(
    run_dir: Path,
    *,
    model: str = "mufu",
    split_id: str = DEFAULT_SPLIT_ID,
    promoted_path: Path | None = None,
) -> bool:
    prom = promoted_path or PROMOTED_PATH
    if not prom.exists():
        return False
    promoted = json.loads(prom.read_text())
    return promoted.get(model, {}).get(split_id) == run_dir.name


def missing_promoted_runs(
    promoted_path: Path | None = None,
    runs_dir: Path | None = None,
) -> list[str]:
    """Promoted `run_id`s in `promoted.json` missing a local `run.json` in `runs_dir`."""
    if promoted_path is None:
        promoted_path = PROMOTED_PATH
    if runs_dir is None:
        runs_dir = RUNS_DIR
    if not promoted_path.exists():
        return [promoted_path.name]
    promoted = json.loads(promoted_path.read_text())
    run_ids = [
        str(run_id)
        for key, mapping in promoted.items()
        if key != "tag" and isinstance(mapping, dict)
        for run_id in mapping.values()
    ]
    if not run_ids:
        return [promoted_path.name]
    return [r for r in run_ids if not (runs_dir / r / "run.json").exists()]


def rebuild_if_promoted_present(
    promoted_path: Path | None = None,
    runs_dir: Path | None = None,
) -> None:
    missing = missing_promoted_runs(promoted_path, runs_dir)
    if not missing:
        rebuild_tracked_results(promoted_path=promoted_path, runs_dir=runs_dir)
    else:
        print(
            f"Skipped rebuilding data/baseline/*.csv: missing promoted runs {missing} "
            "in runs/ (run `python drivers/download_mufu.py`, then "
            "`python drivers/rebuild_results.py`)."
        )


def promote_run(
    run_dir: Path,
    *,
    tag: str | None = None,
    promoted_path: Path | None = None,
    runs_dir: Path | None = None,
    results_path: Path = RESULTS_PATH,
    data_path: Path = COMBINED_CSV_PATH,
    rebuild: bool = True,
    n_boot: int = N_BOOT,
) -> pd.DataFrame | None:
    """Register `run_dir` in `promoted.json` and optionally rebuild `data/baseline/*.csv`."""
    if promoted_path is None:
        promoted_path = PROMOTED_PATH
    if runs_dir is None:
        runs_dir = RUNS_DIR
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing run.json in {run_dir}")
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "complete":
        raise ValueError(
            f"refusing to promote incomplete run {run_dir.name} "
            f"(status={meta.get('status')!r})"
        )
    seeds = [int(s) for s in meta.get("seeds", SEEDS)]
    if list(meta.get("completed_seeds", seeds)) != seeds:
        raise ValueError(
            f"refusing to promote run {run_dir.name}: completed_seeds "
            f"{meta.get('completed_seeds')} != {seeds}"
        )
    for seed in seeds:
        for split in ("val", "test"):
            p = run_dir / "predictions" / f"{split}_seed_{seed}.csv"
            if not p.exists():
                raise FileNotFoundError(
                    f"refusing to promote {run_dir.name}: missing {p}"
                )

    prev_promoted_bytes = promoted_path.read_bytes() if promoted_path.exists() else None
    if prev_promoted_bytes is not None:
        promoted = json.loads(prev_promoted_bytes.decode("utf-8"))
    else:
        promoted = {
            "tag": tag or "v1.0.0",
            "mufu": {},
            "rf_bioactivity": {},
            "rf_toxicity": {},
        }
    if tag is not None:
        promoted["tag"] = tag
    model = str(meta["model"])
    split_id = str(meta["split_id"])
    promoted.setdefault(model, {})[split_id] = run_dir.name
    promoted_path.parent.mkdir(parents=True, exist_ok=True)
    promoted_path.write_text(json.dumps(promoted, indent=2) + "\n")

    if not rebuild:
        return None
    try:
        return rebuild_tracked_results(
            promoted_path=promoted_path,
            runs_dir=runs_dir,
            results_path=results_path,
            data_path=data_path,
            n_boot=n_boot,
        )
    except BaseException:
        if prev_promoted_bytes is None:
            promoted_path.unlink(missing_ok=True)
        else:
            promoted_path.write_bytes(prev_promoted_bytes)
        raise


def rebuild_tracked_results(
    *,
    promoted_path: Path | None = None,
    runs_dir: Path | None = None,
    results_path: Path = RESULTS_PATH,
    paired_path: Path | None = None,
    similarity_path: Path | None = None,
    data_path: Path = COMBINED_CSV_PATH,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """Recompute `data/baseline/*.csv` and per-run metric CSVs from promoted runs' saved predictions."""
    from baseline import compare, similarity

    if promoted_path is None:
        promoted_path = PROMOTED_PATH
    if runs_dir is None:
        runs_dir = RUNS_DIR
    if not promoted_path.exists():
        raise FileNotFoundError(f"missing promoted.json at {promoted_path}")
    promoted = json.loads(promoted_path.read_text())

    splits_by_id: dict[str, pd.DataFrame] = {}

    def get_split(split_id: str) -> pd.DataFrame:
        if split_id not in splits_by_id:
            splits_by_id[split_id] = load_benchmark_split(data_path, split_id)
        return splits_by_id[split_id]

    frames: list[pd.DataFrame] = []
    default_run_dirs: list[Path] = []
    model_keys = [k for k in promoted if k != "tag" and isinstance(promoted[k], dict)]

    for model in sorted(model_keys):
        for split_id, run_id in sorted(promoted[model].items()):
            run_dir = runs_dir / run_id
            df = get_split(split_id)
            run_df = score_run_predictions(
                run_dir, df, include_test=True, n_boot=n_boot, data_path=data_path
            )
            frames.append(run_df)
            if split_id == DEFAULT_SPLIT_ID:
                default_run_dirs.append(run_dir)
            if model == "mufu":
                from mufu.experiment import score_mufu_run_test

                score_mufu_run_test(run_dir, df)

    if not frames:
        raise ValueError(f"no promoted runs listed in {promoted_path}")

    all_rows = (
        pd.concat(frames, ignore_index=True)
        .reindex(columns=COLUMNS)
        .sort_values(SORT_KEYS, ignore_index=True)
    )
    shas = sorted(all_rows["data_sha256"].dropna().unique())
    if len(shas) > 1:
        raise ValueError(f"promoted runs disagree on data_sha256: {shas}")

    pair_out_path = paired_path or results_path.with_name("results_paired.csv")
    paired_df: pd.DataFrame | None = None
    mufu_dir: Path | None = None
    mufu_default_id = promoted.get("mufu", {}).get(DEFAULT_SPLIT_ID)
    if mufu_default_id:
        mufu_dir = runs_dir / mufu_default_id
        default_df = get_split(DEFAULT_SPLIT_ID)
        paired_parts = [
            compare.compare_run_against_baselines(
                mufu_dir,
                default_df,
                split=split,
                promoted_path=promoted_path,
                runs_dir=runs_dir,
                include_promoted_mufu=False,
                n_boot=n_boot,
                save=False,
            )
            for split in ("val", "test")
        ]
        if any(not p.empty for p in paired_parts):
            paired_df = pd.concat(paired_parts, ignore_index=True)

    sim_out_path = similarity_path or results_path.with_name(
        "results_by_similarity.csv"
    )
    sim_df: pd.DataFrame | None = None
    if default_run_dirs:
        default_df = get_split(DEFAULT_SPLIT_ID)
        if (default_df["split"] == "train").any():
            sim_df = similarity.similarity_table(default_df, default_run_dirs)

    write_results(all_rows, results_path)
    if paired_df is not None and mufu_dir is not None:
        paired_df.to_csv(pair_out_path, index=False, lineterminator="\n")
        val_paired_df = paired_df[paired_df["split"] == "val"]
        val_paired_df.to_csv(
            mufu_dir / "val_paired.csv", index=False, lineterminator="\n"
        )
    if sim_df is not None:
        sim_df.to_csv(sim_out_path, index=False, lineterminator="\n")

    return all_rows
