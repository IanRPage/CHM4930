"""
Record baseline metrics in the tracked CSV `ai/data/baseline/results.csv`.

Each row is one measurement, tagged with the snapshot hash and git commit it came from.
Re-running a baseline replaces that model's rows.
"""

import datetime
import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.cache import write_csv
from pipeline.combined_loader import COMBINED_CSV_PATH

RESULTS_PATH = COMBINED_CSV_PATH.with_name("results.csv")
COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split",
    "metric",
    "value",
    "n_labeled",
    "n_positive",
    "params",
    "data_sha256",
    "git_commit",
    "date",
]
SORT_KEYS = ["model", "dataset", "endpoint", "split", "metric"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
) -> pd.DataFrame:
    new = pd.DataFrame(rows).assign(
        model=model,
        data_sha256=file_sha256(data_path),
        git_commit=git_commit(),
        date=datetime.datetime.now(datetime.UTC).date().isoformat(),
    )
    new = new.reindex(columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["model"] != model], new], ignore_index=True)
    out = new[COLUMNS].sort_values(SORT_KEYS, ignore_index=True)
    write_csv(out, path)
    return out
