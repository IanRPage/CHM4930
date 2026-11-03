"""
Turn the combined table into MuFu training batches. Molecules are featurized once when
the dataset is built (not every epoch). A label's mask is True where the label exists.
Rows RDKit can't parse are dropped.

From Python:

    datasets = split_datasets(scaffold_split(load_combined()))
    loader = DataLoader(
        datasets["train"], batch_size=64, shuffle=True, drop_last=True, collate_fn=collate
    )

drop_last avoids a final batch of one molecule (GNNEncoder's BatchNorm rejects in
training mode).
"""

import copy

import numpy as np
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
from pipeline.split import SPLITS, _scaffold_groups, _source_flags, murcko_scaffold

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


def to_device(
    batch: MuFuBatch, device: torch.device | str, non_blocking: bool = False
) -> MuFuBatch:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, Batch):
            moved[k] = copy.copy(v).to(device, non_blocking=non_blocking)
        elif isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, list):
            moved[k] = list(v)
        else:
            moved[k] = v
    return moved


def _split_val_by_scaffold(val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition `val` 50/50 by Bemis-Murcko scaffold into `(val_select, val_calib)`."""
    if len(val_df) < 2:
        return val_df, val_df
    scaffolds = (
        val_df["scaffold"].fillna("").tolist()
        if "scaffold" in val_df.columns
        else [murcko_scaffold(s) for s in val_df["smiles"]]
    )
    groups = _scaffold_groups(val_df["smiles"].tolist(), scaffolds)
    if len(groups) < 2:
        return val_df, val_df
    flags = _source_flags(val_df)
    member = (
        val_df[flags].fillna(False).to_numpy(bool)
        if flags
        else np.ones((len(val_df), 1), dtype=bool)
    )
    budget = 0.5 * member.sum(axis=0)
    counts = np.zeros(member.shape[1])
    in_select = np.zeros(len(val_df), dtype=bool)
    for group in groups:
        group_counts = member[group].sum(axis=0)
        if np.all(counts + group_counts <= budget):
            counts += group_counts
            in_select[group] = True
    if not in_select.any() or in_select.all():
        in_select[:] = False
        in_select[groups[0]] = True
    return val_df.iloc[in_select], val_df.iloc[~in_select]


def split_datasets(df: pd.DataFrame) -> dict[str, MuFuDataset]:
    datasets = {name: MuFuDataset(df[df["split"] == name]) for name in SPLITS}
    val_select_df, val_calib_df = _split_val_by_scaffold(df[df["split"] == "val"])
    datasets["val_select"] = MuFuDataset(val_select_df)
    datasets["val_calib"] = MuFuDataset(val_calib_df)
    return datasets
