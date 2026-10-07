"""
Turn the combined table into MuFu training batches. Molecules are featurized once when
the dataset is built (not every epoch). A label's mask is True where the label exists.
Rows RDKit can't parse are dropped.

From Python:

    datasets = split_datasets(scaffold_split(load_combined()))
    loader = DataLoader(datasets["train"], batch_size=64, shuffle=True, collate_fn=collate)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from mufu.contract import (
    ACTIVE_COLUMNS,
    FINGERPRINT_BITS,
    LABEL_FILL,
    PIC50_COLUMNS,
    TOX_COLUMNS,
    MuFuBatch,
)
from pipeline.preprocess import featurize_many
from pipeline.split import SPLITS

LABELS = {
    "y_pic50": PIC50_COLUMNS,
    "y_active": ACTIVE_COLUMNS,
    "y_tox": TOX_COLUMNS,
}
MASKS = {"mask_bio": PIC50_COLUMNS, "mask_tox": TOX_COLUMNS}


class MuFuDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        graphs, failed = featurize_many(df["smiles"], FINGERPRINT_BITS)
        df = df.drop(index=failed)

        self.smiles = [g.smiles for g in graphs]
        self.fingerprints = (
            torch.cat([g.fp for g in graphs])
            if graphs
            else torch.empty(0, FINGERPRINT_BITS, dtype=torch.uint8)
        )
        self.graphs = [
            Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in graphs
        ]
        self.labels = {
            key: torch.tensor(
                df[list(cols)].fillna(LABEL_FILL).to_numpy(), dtype=torch.float
            )
            for key, cols in LABELS.items()
        }
        self.masks = {
            key: torch.tensor(df[list(cols)].notna().to_numpy(), dtype=torch.bool)
            for key, cols in MASKS.items()
        }
        flags = [c for c in df.columns if c.startswith("in_")]
        self.sources = df[flags].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, i: int) -> dict:
        return {
            "smiles": self.smiles[i],
            "fingerprint": self.fingerprints[i],
            "graph": self.graphs[i],
            **{key: value[i] for key, value in {**self.labels, **self.masks}.items()},
        }


def collate(examples: list[dict]) -> MuFuBatch:
    batch = {
        "smiles": [e["smiles"] for e in examples],
        "fingerprint": torch.stack([e["fingerprint"] for e in examples]).float(),
        "graph": Batch.from_data_list([e["graph"] for e in examples]),
    }
    for key in (*LABELS, *MASKS):
        batch[key] = torch.stack([e[key] for e in examples])
    return batch


def split_datasets(df: pd.DataFrame) -> dict[str, MuFuDataset]:
    return {name: MuFuDataset(df[df["split"] == name]) for name in SPLITS}
