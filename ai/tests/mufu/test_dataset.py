import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader

from featurize.fingerprint import mol_to_ecfp4
from mufu import contract as c
from mufu.dataset import MuFuDataset, collate, split_datasets
from pipeline.combined_loader import load_combined


def make_df():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "C", "not a smiles"]
    df = pd.DataFrame(
        np.nan,
        index=range(5),
        columns=[*c.PIC50_COLUMNS, *c.ACTIVE_COLUMNS, *c.TOX_COLUMNS],
    )
    df.insert(0, "smiles", smiles)
    df.loc[0, [c.PIC50_COLUMNS[0], c.ACTIVE_COLUMNS[0]]] = [7.5, 1.0]
    df.loc[1, [c.TOX_COLUMNS[0], c.TOX_COLUMNS[3]]] = [1.0, 0.0]
    df.loc[2, [c.PIC50_COLUMNS[1], c.ACTIVE_COLUMNS[1], c.TOX_COLUMNS[-1]]] = [
        5.0,
        0.0,
        1.0,
    ]
    df.loc[3, c.TOX_COLUMNS[1]] = 0.0
    df.loc[4, c.TOX_COLUMNS[1]] = 1.0
    df["in_bace1"] = [True, False, False, False, False]
    df["in_tox21"] = [False, True, False, True, True]
    return df


def test_batch_passes_contract():
    ds = MuFuDataset(make_df())
    for batch in DataLoader(ds, batch_size=3, collate_fn=collate):
        c.validate_batch(batch)


def test_unparseable_row_dropped_and_aligned():
    ds = MuFuDataset(make_df())
    assert len(ds) == 4
    assert ds.smiles == ["CCO", "c1ccccc1", "CC(=O)O", "C"]
    assert ds.labels["y_tox"][3, 1] == 0.0
    assert len(ds.sources) == 4


def test_masks_match_nans_and_fill():
    ds = MuFuDataset(make_df())
    batch = collate([ds[i] for i in range(len(ds))])
    assert batch["mask_bio"].tolist() == [
        [True, False],
        [False, False],
        [False, True],
        [False, False],
    ]
    assert batch["mask_tox"].sum().item() == 4
    assert batch["mask_tox"][1, 0] and batch["mask_tox"][1, 3]
    assert not batch["y_pic50"].isnan().any()
    assert (batch["y_pic50"][~batch["mask_bio"]] == c.LABEL_FILL).all()
    assert (batch["y_tox"][~batch["mask_tox"]] == c.LABEL_FILL).all()
    assert batch["y_pic50"][0, 0] == 7.5


def test_features_match_featurizers():
    ds = MuFuDataset(make_df())
    batch = collate([ds[0], ds[3]])
    expected = mol_to_ecfp4(Chem.MolFromSmiles("CCO"))
    assert batch["fingerprint"][0].tolist() == expected.tolist()
    assert batch["graph"].num_graphs == 2
    assert batch["graph"].x.shape[0] == 3 + 1


def test_split_datasets_sizes():
    df = make_df().iloc[:4].assign(split=["train", "train", "val", "test"])
    sets = split_datasets(df)
    assert {k: len(v) for k, v in sets.items()} == {"train": 2, "val": 1, "test": 1}


def test_sources_follow_dataset_order():
    ds = MuFuDataset(make_df())
    assert ds.sources["in_bace1"].tolist() == [True, False, False, False]


def test_real_snapshot_head():
    ds = MuFuDataset(load_combined().head(64))
    batch = collate([ds[i] for i in range(len(ds))])
    c.validate_batch(batch)
    assert (batch["mask_bio"].any(1) | batch["mask_tox"].any(1)).all()


def test_empty_sources_when_no_flags():
    ds = MuFuDataset(make_df().drop(columns=["in_bace1", "in_tox21"]))
    assert ds.sources.shape[1] == 0


def test_empty_table():
    ds = MuFuDataset(make_df().iloc[:0])
    assert len(ds) == 0
    assert ds.fingerprints.shape == (0, c.FINGERPRINT_BITS)


def test_duplicate_index_stays_aligned():
    df = make_df().set_axis([0, 0, 1, 1, 1])
    ds = MuFuDataset(df)
    assert len(ds) == len(ds.sources) == ds.labels["y_tox"].shape[0] == 4
