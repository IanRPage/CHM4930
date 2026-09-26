"""
Download, clean, and load Tox21 and ClinTox toxicity data from MoleculeNet.

How to use as CLI tool (from `ai/`, with `PYTHONPATH=src`):

    python -m pipeline.toxicity_loader                     # use cached data, download if missing
    python -m pipeline.toxicity_loader --refresh           # re-download from MoleculeNet
    python -m pipeline.toxicity_loader --dataset clintox   # only one dataset (default: all)

Prints the molecule count and per-task label summary for each dataset. From a
python script, use `load_toxicity_data()`.

`load_toxicity_data()` mirrors deepchem's `load_tox21`/`load_clintox` output:
`(tasks, (DiskDataset,), transformers)`. The data is pulled unsplit with SMILES
as the representation and no transformers. Missing labels have weight `w == 0`.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from deepchem.data import NumpyDataset
from deepchem.data.datasets import DiskDataset
from deepchem.feat import RawFeaturizer
from deepchem.molnet.load_function.clintox_datasets import load_clintox
from deepchem.molnet.load_function.tox21_datasets import load_tox21
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from rdkit import RDLogger

from pipeline.preprocess import standardize_smiles

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DUPLICATE_MODES = {"tox21": "keep_first", "clintox": "remove_all"}
DATASETS = list(DUPLICATE_MODES)

_TASKS_FILE = "tasks.json"
_TRANSFORMERS_FILE = "transformers.joblib"


def clean_dataset(toxicity_data, duplicate_mode):
    valid_indices = []
    standardized_smiles = []

    # maps standardized SMILES -> list of original dataset indices
    smiles_to_indices = {}

    for i, smiles in enumerate(toxicity_data.ids):
        standardized = standardize_smiles(smiles)

        if standardized is None:
            continue

        if standardized not in smiles_to_indices:
            smiles_to_indices[standardized] = []

        smiles_to_indices[standardized].append(i)

    for standardized, indices in smiles_to_indices.items():
        # keeping first duplicate instance for tox21
        if len(indices) == 1 or duplicate_mode == "keep_first":
            valid_indices.append(indices[0])
            standardized_smiles.append(standardized)

        # remove all duplicated for clintox
        elif duplicate_mode == "remove_all":
            continue

    toxicity_data_filtered = toxicity_data.select(valid_indices)

    toxicity_data_filtered = NumpyDataset(
        X=standardized_smiles,
        y=toxicity_data_filtered.y,
        w=toxicity_data_filtered.w,
        ids=standardized_smiles,
    )

    return toxicity_data_filtered


def _fetch(dataset_name: str):
    load = load_tox21 if dataset_name == "tox21" else load_clintox
    return load(
        splitter=None,
        featurizer=RawFeaturizer(smiles=True),
        transformers=[],
        reload=False,
    )


def _is_complete(cache_path: Path) -> bool:
    return (cache_path / _TASKS_FILE).exists() and (
        cache_path / _TRANSFORMERS_FILE
    ).exists()


def download_toxicity_data(dataset_name: str, cache_path: Path):
    tasks, (raw,), transformers = _fetch(dataset_name)
    cleaned = clean_dataset(raw, DUPLICATE_MODES[dataset_name])
    log.info("cleaned %s: %d raw -> %d molecules", dataset_name, len(raw), len(cleaned))

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    try:
        DiskDataset.from_numpy(
            X=cleaned.X,
            y=cleaned.y,
            w=cleaned.w,
            ids=cleaned.ids,
            tasks=tasks,
            data_dir=str(tmp_path),
        )
        with open(tmp_path / _TASKS_FILE, "w") as f:
            json.dump(list(tasks), f)
        save_to_disk(transformers, str(tmp_path / _TRANSFORMERS_FILE))

        shutil.rmtree(cache_path, ignore_errors=True)
        tmp_path.rename(cache_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
    log.info("wrote %d molecules to %s", len(cleaned), cache_path)


def load_toxicity_data(
    dataset_name: str, refresh: bool = False, data_dir: Path = DATA_DIR
):
    if dataset_name not in DUPLICATE_MODES:
        raise ValueError(
            f"unknown dataset {dataset_name!r}, expected one of {DATASETS}"
        )

    cache_path = data_dir / f"MolNet-{dataset_name}"
    if refresh or not _is_complete(cache_path):
        if cache_path.exists() and not refresh:
            log.warning("cache %s is incomplete, re-downloading", cache_path)
        log.info("downloading %s data to %s", dataset_name, cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        download_toxicity_data(dataset_name, cache_path)
    else:
        log.info("using cached %s", cache_path)

    with open(cache_path / _TASKS_FILE) as f:
        tasks = json.load(f)
    dataset = DiskDataset(str(cache_path))
    transformers = load_from_disk(str(cache_path / _TRANSFORMERS_FILE))
    return (tasks, (dataset,), transformers)


def summarize(tasks: list[str], dataset) -> pd.DataFrame:
    labeled = dataset.w != 0
    positives = (dataset.y == 1) & labeled
    n_labeled = labeled.sum(axis=0)
    return pd.DataFrame(
        {
            "labeled": n_labeled,
            "missing": (~labeled).sum(axis=0),
            "positive_rate": np.divide(
                positives.sum(axis=0),
                n_labeled,
                out=np.full(len(tasks), np.nan),
                where=n_labeled > 0,
            ),
        },
        index=pd.Index(tasks, name="task"),
    )


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
    logging.getLogger("deepchem").setLevel(logging.ERROR)

    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        tasks, (dataset,), _ = load_toxicity_data(name, refresh=args.refresh)
        print(f"\n{name}: {len(dataset)} molecules")
        print(summarize(tasks, dataset).round(3))


if __name__ == "__main__":
    main()
