"""Download, clean, and load bioactivity data from ChEMBL.

How to use as CLI tool (from `ai/`):

    python src/bioactivity_loader.py                  # use cached CSV, download if missing
    python src/bioactivity_loader.py --refresh        # re-download from ChEMBL
    python src/bioactivity_loader.py --threshold 7.0  # pIC50 cutoff for "active" (default 6.0)

Prints the molecule count and a pIC50 summary. From a python script, use
`load_bace1()`.
"""

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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

# BACE-1 specific constants
BACE1_TARGET_ID = "CHEMBL4822"
BACE1_CSV_PATH = DATA_DIR / "chembl-bace-1.csv"
PIC50_ACTIVE_THRESHOLD = 6.0
CSV_COLUMNS = ["molecule_chembl_id", "smiles", "pIC50", "n_meas", "pIC50_spread"]

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


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


def fetch_activities() -> pd.DataFrame:
    params = {
        "target_chembl_id": BACE1_TARGET_ID,
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
        raise RuntimeError(f"ChEMBL returned no IC50 rows for {BACE1_TARGET_ID}")
    if len(rows) != total_count:
        raise RuntimeError(
            f"ChEMBL reported {total_count} IC50 rows for {BACE1_TARGET_ID} "
            f"but only {len(rows)} were retrieved"
        )
    return pd.DataFrame.from_records(rows)


# strips salts/counterions by keeping the largest fragment (charges and stereo preserved)
def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _LARGEST_FRAGMENT.choose(mol)
    return Chem.MolToSmiles(mol)


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

    df = df.rename(columns={"pchembl_value": "pIC50"}).assign(
        smiles=lambda d: d["canonical_smiles"].map(standardize_smiles)
    )
    n_unparsed = int(df["smiles"].isna().sum())
    if n_unparsed:
        log.warning("dropping %d rows whose SMILES RDKit couldn't parse", n_unparsed)
        df = df.dropna(subset=["smiles"])

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


def download_bace1(csv_path: Path = BACE1_CSV_PATH) -> None:
    cleaned = clean_activities(fetch_activities())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    cleaned.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)
    log.info("wrote %d molecules to %s", len(cleaned), csv_path)


def preprocess(
    df: pd.DataFrame, threshold: float = PIC50_ACTIVE_THRESHOLD
) -> pd.DataFrame:
    out = df.copy()
    out["active"] = (out["pIC50"] >= threshold).astype(int)
    return out


def load_bace1(
    threshold: float = PIC50_ACTIVE_THRESHOLD,
    csv_path: Path = BACE1_CSV_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    if refresh or not csv_path.exists():
        log.info("downloading BACE-1 data to %s", csv_path)
        download_bace1(csv_path)
    else:
        log.info("using cached %s", csv_path)

    df = pd.read_csv(csv_path)
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {sorted(missing)}; "
            "delete it or re-run with refresh=True / --refresh"
        )
    return preprocess(df, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-download even if the CSV exists"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PIC50_ACTIVE_THRESHOLD,
        help="pIC50 cutoff for the active label (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_bace1(threshold=args.threshold, refresh=args.refresh)
    print(f"\n{len(df)} molecules")
    print(df["pIC50"].describe().round(3))


if __name__ == "__main__":
    main()
