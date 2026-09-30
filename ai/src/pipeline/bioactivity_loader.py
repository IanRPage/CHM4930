"""
Download, clean, and load bioactivity data from ChEMBL.

How to use as CLI tool (from `ai/`, with `PYTHONPATH=src`):

    python -m pipeline.bioactivity_loader                  # use cached CSVs, download if missing
    python -m pipeline.bioactivity_loader --refresh        # re-download from ChEMBL
    python -m pipeline.bioactivity_loader --dataset egfr   # only one dataset (default: all)
    python -m pipeline.bioactivity_loader --threshold 7.0  # pIC50 cutoff for "active" (default 6.0)

Prints the molecule count, active label summary, and pIC50 summary for each dataset. From
Python, use `load_bioactivity_data("bace1" or "egfr")` for pIC50 and active labels;
`load_bace1()` and `load_egfr()` are shortcuts for it.
"""

import argparse
import json
import logging
import urllib.request
from functools import partial
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from pipeline.cache import DATA_DIR, load_csv, write_csv
from pipeline.preprocess import standardize_smiles_column, summarize_labels

log = logging.getLogger(__name__)

CHEMBL_HOST = "https://www.ebi.ac.uk"
ACTIVITY_URL = f"{CHEMBL_HOST}/chembl/api/data/activity.json"
PAGE_SIZE = 1000
REQUEST_TIMEOUT_S = 60
API_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "pchembl_value",
    "standard_relation",
    "standard_units",
    "assay_type",
    "potential_duplicate",
    "data_validity_comment",
]

# Target IDs and cache paths
BACE1_TARGET_ID = "CHEMBL4822"
EGFR_TARGET_ID = "CHEMBL203"
TARGETS = {
    "bace1": {"target_id": BACE1_TARGET_ID, "csv_name": "chembl-bace-1.csv"},
    "egfr": {"target_id": EGFR_TARGET_ID, "csv_name": "chembl-egfr.csv"},
}
BACE1_CSV_PATH = DATA_DIR / TARGETS["bace1"]["csv_name"]
EGFR_CSV_PATH = DATA_DIR / TARGETS["egfr"]["csv_name"]
PIC50_ACTIVE_THRESHOLD = 6.0
CSV_COLUMNS = ["molecule_chembl_id", "smiles", "pIC50", "n_meas", "pIC50_spread"]


# validate one ChEMBL activity page, returns (records, next path, total count)
def _parse_page(payload: dict) -> tuple[list[dict], str | None, int]:
    activities = payload.get("activities")
    page_meta = payload.get("page_meta")
    if not isinstance(activities, list) or not isinstance(page_meta, dict):
        raise TypeError("unexpected ChEMBL response: missing activities or page_meta")

    total_count = page_meta.get("total_count")
    next_path = page_meta.get("next")
    if not isinstance(total_count, int):
        raise TypeError("unexpected ChEMBL response: page_meta has no total_count")
    if next_path is not None and not (
        isinstance(next_path, str) and next_path.startswith("/")
    ):
        raise RuntimeError(f"unexpected ChEMBL response: bad next link {next_path!r}")

    for record in activities:
        missing = [field for field in API_FIELDS if field not in record]
        if missing:
            raise RuntimeError(f"ChEMBL record is missing fields {missing}")
    return activities, next_path, total_count


def fetch_activities(target_id: str = BACE1_TARGET_ID) -> pd.DataFrame:
    params = {
        "target_chembl_id": target_id,
        "standard_type": "IC50",
        "limit": PAGE_SIZE,
    }
    url = f"{ACTIVITY_URL}?{urlencode(params)}"

    rows = []
    total_count = None
    while url:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
            payload = json.load(response)
        activities, next_path, page_total = _parse_page(payload)
        if total_count is None:
            total_count = page_total
        rows.extend(
            {field: record[field] for field in API_FIELDS} for record in activities
        )
        log.info("retrieved %d records", len(rows))
        url = f"{CHEMBL_HOST}{next_path}" if next_path else None

    if not rows:
        raise RuntimeError(f"ChEMBL returned no IC50 rows for {target_id}")
    if len(rows) != total_count:
        raise RuntimeError(
            f"ChEMBL reported {total_count} IC50 rows for {target_id} "
            f"but only {len(rows)} were retrieved"
        )
    return pd.DataFrame.from_records(rows)


# filters to exact binding measurements, then collapses to one row per molecule
def clean_activities(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.assign(pchembl_value=pd.to_numeric(raw["pchembl_value"], errors="coerce"))
    log.info("%-40s %6d rows", "raw activities", len(df))

    # log shows which filter removes how much
    steps = [
        ("standard_relation == '='", lambda d: d["standard_relation"] == "="),
        ("standard_units == 'nM'", lambda d: d["standard_units"] == "nM"),
        ("assay_type == 'B' (binding)", lambda d: d["assay_type"] == "B"),
        ("potential_duplicate == 0", lambda d: d["potential_duplicate"] == 0),
        ("no data_validity_comment", lambda d: d["data_validity_comment"].isna()),
        (
            "has SMILES and pChEMBL",
            lambda d: d["canonical_smiles"].notna() & d["pchembl_value"].notna(),
        ),
    ]
    for label, keep in steps:
        df = df[keep(df)]
        log.info("%-40s %6d rows", f"after {label}", len(df))

    df = standardize_smiles_column(
        df.rename(columns={"pchembl_value": "pIC50"}), column="canonical_smiles"
    )

    molecules = df.groupby("smiles", as_index=False).agg(
        molecule_chembl_id=("molecule_chembl_id", "min"),
        pIC50=("pIC50", "median"),
        n_meas=("pIC50", "size"),
        pIC50_spread=("pIC50", lambda s: s.max() - s.min()),
    )
    log.info(
        "%-40s %6d molecules (%d with repeat measurements)",
        "aggregated to one row per structure",
        len(molecules),
        int((molecules["n_meas"] > 1).sum()),
    )
    return molecules[CSV_COLUMNS].sort_values("molecule_chembl_id", ignore_index=True)


def download_bioactivity_data(target: str, csv_path: Path) -> None:
    raw = fetch_activities(TARGETS[target]["target_id"])
    write_csv(clean_activities(raw), csv_path)


def add_active_label(
    df: pd.DataFrame, threshold: float = PIC50_ACTIVE_THRESHOLD
) -> pd.DataFrame:
    out = df.copy()
    out["active"] = (out["pIC50"] >= threshold).astype(int)
    return out


def _load(target: str, csv_path: Path, threshold: float, refresh: bool) -> pd.DataFrame:
    download = partial(download_bioactivity_data, target)
    df = load_csv(csv_path, download, CSV_COLUMNS, refresh)
    return add_active_label(df, threshold)


def load_bioactivity_data(
    target: str,
    threshold: float = PIC50_ACTIVE_THRESHOLD,
    refresh: bool = False,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}, expected one of {list(TARGETS)}")
    csv_path = data_dir / TARGETS[target]["csv_name"]
    return _load(target, csv_path, threshold, refresh)


def load_bace1(
    threshold: float = PIC50_ACTIVE_THRESHOLD,
    csv_path: Path = BACE1_CSV_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    return _load("bace1", csv_path, threshold, refresh)


def load_egfr(
    threshold: float = PIC50_ACTIVE_THRESHOLD,
    csv_path: Path = EGFR_CSV_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load EGFR pIC50 and derive an active label at the chosen cutoff."""
    return _load("egfr", csv_path, threshold, refresh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-download even if the CSV exists"
    )
    parser.add_argument(
        "--dataset",
        choices=[*TARGETS, "all"],
        default="all",
        help="which dataset to load (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PIC50_ACTIVE_THRESHOLD,
        help="pIC50 cutoff for the active label (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    names = list(TARGETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        df = load_bioactivity_data(name, args.threshold, args.refresh)
        print(f"\n{name}: {len(df)} molecules")
        print(summarize_labels(df, ["active"]).round(3), end="\n\n")
        print(df["pIC50"].describe().round(3))


if __name__ == "__main__":
    main()
