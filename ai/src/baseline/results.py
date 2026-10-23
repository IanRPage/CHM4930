"""
Record benchmark metrics in the tracked CSV `ai/data/baseline/results.csv`.

Each row is one seed's measurement, tagged with the snapshot hash and git commit it came
from. Readers aggregate across seeds with `summarize_seeds()`. Re-running replaces the
`(model, encoder, seed)` rows being written and refuses to drop other models' rows from
an older snapshot unless `allow_drop_stale=True`.
"""

import datetime
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.cache import write_csv
from pipeline.combined_loader import COMBINED_CSV_PATH

log = logging.getLogger(__name__)

RESULTS_PATH = COMBINED_CSV_PATH.with_name("results.csv")
COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split",
    "seed",
    "metric",
    "value",
    "n_labeled",
    "n_positive",
    "params",
    "data_sha256",
    "git_commit",
    "date",
]
SORT_KEYS = ["model", "encoder", "dataset", "endpoint", "split", "seed", "metric"]
SUMMARY_GROUP_KEYS = [
    c for c in COLUMNS if c not in ("seed", "value", "git_commit", "date")
]
SUMMARY_COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split",
    "metric",
    "value",
    "std",
    "n_seeds",
    "n_labeled",
    "n_positive",
    "params",
    "data_sha256",
]


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
        dirty = git("status", "--porcelain", "--untracked-files=no")
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
) -> pd.DataFrame:
    sha = file_sha256(data_path)
    new = pd.DataFrame(rows).assign(
        model=model,
        data_sha256=sha,
        git_commit=git_commit(),
        date=datetime.datetime.now(datetime.UTC).date().isoformat(),
    )
    new = new.reindex(columns=COLUMNS)
    new["seed"] = new["seed"].astype("Int64")
    if path.exists():
        old = pd.read_csv(path).reindex(columns=COLUMNS)
        old["seed"] = old["seed"].astype("Int64")
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
        replaced = pd.MultiIndex.from_frame(new[["model", "encoder", "seed"]])
        keep = ~stale & ~pd.MultiIndex.from_frame(
            old[["model", "encoder", "seed"]]
        ).isin(replaced)
        new = pd.concat([old[keep], new], ignore_index=True)
    out = new[COLUMNS].sort_values(SORT_KEYS, ignore_index=True)
    out["seed"] = out["seed"].astype("Int64")
    write_csv(out, path)
    return out


def summarize_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    grouped = df.groupby(SUMMARY_GROUP_KEYS, dropna=False, as_index=False).agg(
        value=("value", "mean"),
        std=("value", lambda s: s.std(ddof=1)),
        n_seeds=("value", "count"),
    )
    sort_cols = [k for k in SORT_KEYS if k != "seed"]
    return grouped[SUMMARY_COLUMNS].sort_values(sort_cols, ignore_index=True)
