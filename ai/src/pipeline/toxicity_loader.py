"""
Download, clean, and load Tox21 and ClinTox toxicity data from MoleculeNet.

How to use as CLI tool (from `ai/`):

    python -m pipeline.toxicity_loader                     # use cached data, download if missing
    python -m pipeline.toxicity_loader --refresh           # re-download from MoleculeNet
    python -m pipeline.toxicity_loader --dataset clintox   # only one dataset (default: all)

Prints the molecule count and per-task label summary for each dataset. From a
python script, use `load_toxicity_data()`.

Each dataset is cached as a CSV with a `smiles` column followed by one column
per task. Labels are 0/1, and NaN where the compound wasn't measured.
"""

import argparse
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from pipeline.preprocess import standardize_smiles_column, summarize_labels

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MOLECULENET_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{name}.csv.gz"
)
REQUEST_TIMEOUT_S = 60

TASKS = {
    "tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
}
DUPLICATE_MODES = {"tox21": "keep_first", "clintox": "remove_all"}
DATASETS = list(TASKS)


def fetch_dataset(dataset_name: str) -> pd.DataFrame:
    url = MOLECULENET_URL.format(name=dataset_name)
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
        raw = pd.read_csv(response, compression="gzip")

    missing = {"smiles", *TASKS[dataset_name]} - set(raw.columns)
    if missing:
        raise RuntimeError(f"{url} is missing columns {sorted(missing)}")
    return raw


def clean_dataset(raw: pd.DataFrame, tasks: list[str], duplicate_mode: str):
    df = standardize_smiles_column(raw)

    # keep first dup instance for tox21, remove all dups for clintox
    if duplicate_mode == "keep_first":
        df = df.drop_duplicates(subset="smiles", keep="first")
    elif duplicate_mode == "remove_all":
        df = df.drop_duplicates(subset="smiles", keep=False)
    else:
        raise ValueError(f"unknown duplicate_mode {duplicate_mode!r}")

    return df[["smiles", *tasks]].reset_index(drop=True)


def download_toxicity_data(dataset_name: str, csv_path: Path):
    raw = fetch_dataset(dataset_name)
    cleaned = clean_dataset(raw, TASKS[dataset_name], DUPLICATE_MODES[dataset_name])
    log.info("cleaned %s: %d raw -> %d molecules", dataset_name, len(raw), len(cleaned))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    cleaned.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)
    log.info("wrote %d molecules to %s", len(cleaned), csv_path)


def load_toxicity_data(
    dataset_name: str, refresh: bool = False, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    if dataset_name not in TASKS:
        raise ValueError(
            f"unknown dataset {dataset_name!r}, expected one of {DATASETS}"
        )

    csv_path = data_dir / f"MolNet-{dataset_name}.csv"
    if refresh or not csv_path.exists():
        log.info("downloading %s data to %s", dataset_name, csv_path)
        download_toxicity_data(dataset_name, csv_path)
    else:
        log.info("using cached %s", csv_path)

    df = pd.read_csv(csv_path)
    expected = ["smiles", *TASKS[dataset_name]]
    if list(df.columns) != expected:
        raise ValueError(
            f"{csv_path} has columns {list(df.columns)}, expected {expected}; "
            "delete it or re-run with refresh=True as an arg OR add --refresh "
            "flag if using CLI"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-download even if cached"
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="which dataset to load (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RDLogger.DisableLog("rdApp.*")

    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        df = load_toxicity_data(name, refresh=args.refresh)
        print(f"\n{name}: {len(df)} molecules")
        print(summarize_labels(df, TASKS[name]).round(3), end="\n\n")


if __name__ == "__main__":
    main()
