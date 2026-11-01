"""
Record benchmark metrics in the tracked CSV `ai/data/baseline/results.csv`.

Each row is one seed's measurement on one scaffold split (`split_id`). What is constant
per `(model, encoder, split_id, seed)` run (params, hashes, commits, date) is stored once
in `results_runs.csv` next to it, and `load_results()` joins the two. Re-running replaces
the run rows being written and refuses to drop other models' rows from an older snapshot
unless `allow_drop_stale=True`.
"""

import datetime
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.combined_loader import COMBINED_CSV_PATH

log = logging.getLogger(__name__)

RESULTS_PATH = COMBINED_CSV_PATH.with_name("results.csv")
DEFAULT_SPLIT_ID = "default"
COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split_id",
    "split",
    "seed",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_labeled",
    "n_positive",
    "n_tasks",
    "dropped_tasks",
    "params",
    "data_sha256",
    "split_sha256",
    "git_commit",
    "eval_commit",
    "date",
]
RUN_KEYS = ["model", "encoder", "split_id", "seed"]
RUN_INFO = [
    "params",
    "data_sha256",
    "split_sha256",
    "git_commit",
    "eval_commit",
    "date",
]
RUN_COLUMNS = RUN_KEYS + RUN_INFO
METRIC_COLUMNS = [c for c in COLUMNS if c not in RUN_INFO]
SORT_KEYS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split_id",
    "split",
    "seed",
    "metric",
]
SUMMARY_GROUP_KEYS = [
    c
    for c in COLUMNS
    if c
    not in ("seed", "value", "ci_low", "ci_high", "git_commit", "eval_commit", "date")
]
SUMMARY_COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split_id",
    "split",
    "metric",
    "value",
    "std",
    "ci_low",
    "ci_high",
    "n_seeds",
    "n_labeled",
    "n_positive",
    "n_tasks",
    "dropped_tasks",
    "params",
    "data_sha256",
    "split_sha256",
]
SPLIT_GROUP_KEYS = ["model", "encoder", "dataset", "endpoint", "split", "metric"]
SPLIT_SUMMARY_COLUMNS = [
    *SPLIT_GROUP_KEYS,
    "value",
    "split_std",
    "seed_std",
    "n_splits",
    "n_seeds",
    "data_sha256",
]
LOWER_IS_BETTER = frozenset({"rmse"})


def runs_path(path: Path = RESULTS_PATH) -> Path:
    return path.with_name(path.stem + "_runs.csv")


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """
    Metric rows joined with their run's params, snapshot hash, git commit and date.
    """
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    metrics = pd.read_csv(path)
    runs = pd.read_csv(runs_path(path))
    for df in (metrics, runs):
        df["seed"] = df["seed"].astype("Int64")
    return metrics.merge(runs, on=RUN_KEYS, how="left", validate="many_to_one")[COLUMNS]


def write_results(out: pd.DataFrame, path: Path) -> None:
    """
    Split joined rows into the metric and run files. Both are written to temp files and
    the old files backed up before either is replaced. If a replace fails, the old pair
    is restored, so the two files never disagree.
    """
    runs = out[RUN_COLUMNS].drop_duplicates()
    if runs.duplicated(RUN_KEYS).any():
        raise ValueError("a run has more than one params, sha256, commit or date")
    pairs = [
        (out[METRIC_COLUMNS], path),
        (runs.sort_values(RUN_KEYS, ignore_index=True), runs_path(path)),
    ]
    targets = [target for _, target in pairs]
    tmps = [t.with_name(t.name + ".tmp") for t in targets]
    baks = [t.with_name(t.name + ".bak") for t in targets]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for (df, _), tmp in zip(pairs, tmps):
            df.to_csv(tmp, index=False)
        for target, bak in zip(targets, baks):
            if target.exists():
                shutil.copy2(target, bak)
        try:
            for target, tmp in zip(targets, tmps):
                tmp.replace(target)
        except BaseException:
            for target, bak in zip(targets, baks):
                if bak.exists():
                    bak.replace(target)
                else:
                    target.unlink(missing_ok=True)
            raise
    finally:
        for leftover in (*tmps, *baks):
            leftover.unlink(missing_ok=True)
    log.info("wrote %d rows to %s", len(out), path)


def params_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def file_sha256(path: Path) -> str:
    """Hash a dataset CSV using LF bytes, including in an existing CRLF checkout.

    This matches the committed snapshot and historical results without rewriting
    the file. All bytes other than CRLF line endings remain significant.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_commit() -> str:
    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ":/",
            f":(exclude){RESULTS_PATH}",
            f":(exclude){runs_path()}",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def record_results(
    rows: list[dict],
    model: str,
    path: Path = RESULTS_PATH,
    data_path: Path = COMBINED_CSV_PATH,
    *,
    allow_drop_stale: bool = False,
    commit: str | None = None,
) -> pd.DataFrame:
    """
    `commit` is the commit that trained the model, the current one by default.
    `eval_commit` is always the current one.
    """
    sha = file_sha256(data_path)
    head = git_commit()
    new = pd.DataFrame(rows).assign(
        model=model,
        data_sha256=sha,
        git_commit=head if commit is None else commit,
        eval_commit=head,
        date=datetime.datetime.now(datetime.UTC).date().isoformat(),
    )
    new = new.reindex(columns=COLUMNS)
    new["split_id"] = new["split_id"].fillna(DEFAULT_SPLIT_ID).astype(str)
    new["seed"] = new["seed"].astype("Int64")
    if path.exists():
        old = load_results(path)
        stale = old["data_sha256"] != sha
        stale_others = sorted(
            old.loc[stale & (old["model"] != model), "model"].unique()
        )
        if stale_others:
            if not allow_drop_stale:
                raise ValueError(
                    f"refusing to drop results for {stale_others} from a different "
                    "snapshot; pass allow_drop_stale=True to overwrite"
                )
            log.warning(
                "dropping results for %s: computed on a different snapshot",
                stale_others,
            )
        replaced = pd.MultiIndex.from_frame(new[RUN_KEYS])
        keep = ~stale & ~pd.MultiIndex.from_frame(old[RUN_KEYS]).isin(replaced)
        new = pd.concat([old[keep], new], ignore_index=True)
    out = new[COLUMNS].sort_values(SORT_KEYS, ignore_index=True)
    out["seed"] = out["seed"].astype("Int64")
    write_results(out, path)
    return out


def summarize_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per split: mean and std over seeds, and the mean per-seed bootstrap CI.
    """
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = df.reindex(columns=COLUMNS)
    grouped = df.groupby(SUMMARY_GROUP_KEYS, dropna=False, as_index=False).agg(
        value=("value", "mean"),
        std=("value", lambda s: s.std(ddof=1)),
        ci_low=("ci_low", "mean"),
        ci_high=("ci_high", "mean"),
        n_seeds=("value", "count"),
    )
    sort_cols = [k for k in SORT_KEYS if k != "seed"]
    return grouped[SUMMARY_COLUMNS].sort_values(sort_cols, ignore_index=True)


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    `split_std` is the std of the split means, `seed_std` the mean within-split std.
    """
    if df.empty:
        return pd.DataFrame(columns=SPLIT_SUMMARY_COLUMNS)
    per_split = summarize_seeds(df)
    grouped = per_split.groupby(
        [*SPLIT_GROUP_KEYS, "data_sha256"], dropna=False, as_index=False
    ).agg(
        value=("value", "mean"),
        split_std=("value", lambda s: s.std(ddof=1)),
        seed_std=("std", "mean"),
        n_splits=("split_id", "nunique"),
        n_seeds=("n_seeds", "sum"),
    )
    return grouped[SPLIT_SUMMARY_COLUMNS].sort_values(
        SPLIT_GROUP_KEYS, ignore_index=True
    )
