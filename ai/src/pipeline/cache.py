"""
Cache cleaned datasets as CSVs in `ai/data/`.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(csv_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    log.info("wrote %d rows to %s", len(df), csv_path)


def load_csv(
    csv_path: Path,
    download: Callable[[Path], None],
    columns: list[str],
    refresh: bool = False,
) -> pd.DataFrame:
    if refresh or not csv_path.exists():
        log.info("downloading data to %s", csv_path)
        download(csv_path)
    else:
        log.info("using cached %s", csv_path)

    df = pd.read_csv(csv_path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {sorted(missing)}; delete it or re-run "
            "with refresh=True as an arg OR add --refresh flag if using CLI"
        )
    return df
