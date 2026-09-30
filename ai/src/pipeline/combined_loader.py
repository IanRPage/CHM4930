"""
Combine BACE-1, EGFR, Tox21, and ClinTox into a single training table.

How to use as CLI tool (from `ai/`, with `PYTHONPATH=src`):

    python -m pipeline.combined_loader                  # rebuild from cached source CSVs
    python -m pipeline.combined_loader --refresh        # re-download every source first
    python -m pipeline.combined_loader --threshold 7.0  # pIC50 cutoff for "active" (default 6.0)

Prints the row count, overlap counts between sources, and a label summary. From Python,
use `load_combined()`.
"""

import argparse
import logging
from functools import reduce
from pathlib import Path

import pandas as pd

from pipeline.bioactivity_loader import (
    BACE1_CSV_PATH,
    EGFR_CSV_PATH,
    PIC50_ACTIVE_THRESHOLD,
    load_bace1,
    load_egfr,
)
from pipeline.cache import DATA_DIR, write_csv
from pipeline.preprocess import summarize_labels
from pipeline.toxicity_loader import DATASETS, load_toxicity_data

log = logging.getLogger(__name__)

COMBINED_CSV_PATH = DATA_DIR / "combined.csv"
TARGETS = ["BACE1", "EGFR"]
TOX_TASKS = [*DATASETS["tox21"]["tasks"], *DATASETS["clintox"]["tasks"]]
SOURCE_FLAGS = ["in_bace1", "in_egfr", "in_tox21", "in_clintox"]


def _label_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name in ("bace1", "egfr"):
        df = df[["smiles", "pIC50"]].rename(columns={"pIC50": f"pIC50_{name.upper()}"})
    else:
        df = df[["smiles", *DATASETS[name]["tasks"]]]
    return df.assign(**{f"in_{name}": True})


# outer join on smiles
def combine_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = reduce(
        lambda left, right: left.merge(
            right, on="smiles", how="outer", validate="one_to_one"
        ),
        (_label_columns(name, df) for name, df in sources.items()),
    )
    combined[SOURCE_FLAGS] = combined[SOURCE_FLAGS].notna()
    labels = [c for c in combined.columns if c not in SOURCE_FLAGS]
    return combined[[*labels, *SOURCE_FLAGS]].sort_values("smiles", ignore_index=True)


def add_active_labels(
    df: pd.DataFrame, threshold: float = PIC50_ACTIVE_THRESHOLD
) -> pd.DataFrame:
    out = df.copy()
    for target in TARGETS:
        pic50 = out[f"pIC50_{target}"]
        active = (pic50 >= threshold).astype(float).where(pic50.notna())
        out.insert(
            out.columns.get_loc(f"pIC50_{target}") + 1, f"active_{target}", active
        )
    return out


def load_combined(
    threshold: float = PIC50_ACTIVE_THRESHOLD,
    refresh: bool = False,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    sources = {
        "bace1": load_bace1(csv_path=data_dir / BACE1_CSV_PATH.name, refresh=refresh),
        "egfr": load_egfr(csv_path=data_dir / EGFR_CSV_PATH.name, refresh=refresh),
        "tox21": load_toxicity_data("tox21", refresh=refresh, data_dir=data_dir),
        "clintox": load_toxicity_data("clintox", refresh=refresh, data_dir=data_dir),
    }
    combined = combine_sources(sources)
    write_csv(combined, data_dir / COMBINED_CSV_PATH.name)
    return add_active_labels(combined, threshold)


# pairwise compound counts, diagonal is each source's size
def overlap_counts(df: pd.DataFrame) -> pd.DataFrame:
    flags = df[SOURCE_FLAGS].astype(int)
    return flags.T @ flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-download every source first"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PIC50_ACTIVE_THRESHOLD,
        help="pIC50 cutoff for the active labels (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_combined(threshold=args.threshold, refresh=args.refresh)
    print(f"\ncombined: {len(df)} molecules")
    print(overlap_counts(df), end="\n\n")
    active = [f"active_{t}" for t in TARGETS]
    print(summarize_labels(df, [*active, *TOX_TASKS]).round(3), end="\n\n")


if __name__ == "__main__":
    main()
