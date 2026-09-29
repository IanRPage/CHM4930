import urllib.request

import numpy as np
import pandas as pd
import pytest

from pipeline import combined_loader as cl
from pipeline.bioactivity_loader import BACE1_CSV_PATH, CSV_COLUMNS, EGFR_CSV_PATH
from pipeline.toxicity_loader import DATASETS

TASKS = ["T1", "T2"]
NAN = np.nan


def bio(smiles, pic50):
    return pd.DataFrame(
        {
            "molecule_chembl_id": [f"C{i}" for i in range(len(smiles))],
            "smiles": smiles,
            "pIC50": pic50,
            "n_meas": 1,
            "pIC50_spread": 0.0,
        }
    )[CSV_COLUMNS]


def tox(smiles, labels):
    return pd.DataFrame(labels, columns=TASKS).assign(smiles=smiles)


@pytest.fixture
def small_tasks(monkeypatch):
    monkeypatch.setitem(DATASETS, "tox21", {"tasks": TASKS, "keep": "first"})
    monkeypatch.setitem(DATASETS, "clintox", {"tasks": ["C1"], "keep": False})


@pytest.fixture
def sources(small_tasks):
    return {
        "bace1": bio(["CCO", "CCN"], [7.0, NAN]),
        "egfr": bio(["CCO", "CCC"], [5.99, 6.0]),
        "tox21": tox(["CCO", "CCCC"], [[1, NAN], [0, 1]]),
        "clintox": pd.DataFrame({"smiles": ["CCCC"], "C1": [1]}),
    }


@pytest.fixture
def cache_dir(small_tasks, tmp_path, monkeypatch):
    bio(["CCO", "CCN"], [7.0, NAN]).to_csv(tmp_path / BACE1_CSV_PATH.name, index=False)
    bio(["CCO", "CCC"], [5.99, 6.0]).to_csv(tmp_path / EGFR_CSV_PATH.name, index=False)
    tox(["CCO", "CCCC"], [[1, NAN], [0, 1]]).to_csv(
        tmp_path / "MolNet-tox21.csv", index=False
    )
    pd.DataFrame({"smiles": ["CCCC"], "C1": [1]}).to_csv(
        tmp_path / "MolNet-clintox.csv", index=False
    )

    def boom(*args, **kwargs):
        raise AssertionError("tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    return tmp_path


def test_shared_compound_lands_in_one_row_with_every_label_and_flag(sources):
    row = cl.combine_sources(sources).set_index("smiles").loc["CCO"]
    assert row["pIC50_BACE1"] == 7.0
    assert row["pIC50_EGFR"] == 5.99
    assert row["T1"] == 1
    assert row[["in_bace1", "in_egfr", "in_tox21"]].all()
    assert not row["in_clintox"]


def test_combined_has_one_row_per_smiles_in_the_union(sources):
    combined = cl.combine_sources(sources)
    assert combined["smiles"].is_unique
    assert sorted(combined["smiles"]) == ["CCC", "CCCC", "CCN", "CCO"]


def test_combined_column_order(sources):
    assert cl.combine_sources(sources).columns.tolist() == [
        "smiles",
        "pIC50_BACE1",
        "pIC50_EGFR",
        "T1",
        "T2",
        "C1",
        *cl.SOURCE_FLAGS,
    ]


def test_duplicate_smiles_in_a_source_raises(sources):
    sources["egfr"] = bio(["CCO", "CCO"], [6.0, 7.0])
    with pytest.raises(pd.errors.MergeError):
        cl.combine_sources(sources)


def test_missing_labels_stay_nan(sources):
    combined = cl.combine_sources(sources).set_index("smiles")
    assert np.isnan(combined.loc["CCN", "pIC50_BACE1"])
    assert np.isnan(combined.loc["CCN", "pIC50_EGFR"])
    assert np.isnan(combined.loc["CCO", "T2"])
    assert np.isnan(combined.loc["CCCC", "pIC50_BACE1"])
    assert np.isnan(combined.loc["CCC", "T1"])


def test_flags_are_bool_without_nan(sources):
    flags = cl.combine_sources(sources)[cl.SOURCE_FLAGS]
    assert (flags.dtypes == bool).all()
    assert flags.sum().tolist() == [2, 2, 2, 1]


def test_active_labels_use_the_threshold_and_keep_nan(sources):
    combined = cl.add_active_labels(cl.combine_sources(sources)).set_index("smiles")
    assert combined.loc["CCO", "active_BACE1"] == 1.0
    assert combined.loc["CCO", "active_EGFR"] == 0.0
    assert combined.loc["CCC", "active_EGFR"] == 1.0
    assert np.isnan(combined.loc["CCN", "active_BACE1"])
    assert np.isnan(combined.loc["CCCC", "active_EGFR"])


def test_active_labels_follow_a_custom_threshold(sources):
    combined = cl.add_active_labels(cl.combine_sources(sources), threshold=7.5)
    assert (combined["active_BACE1"].dropna() == 0).all()


def test_active_columns_sit_after_their_pic50(sources):
    columns = cl.add_active_labels(cl.combine_sources(sources)).columns.tolist()
    assert columns[:5] == [
        "smiles",
        "pIC50_BACE1",
        "active_BACE1",
        "pIC50_EGFR",
        "active_EGFR",
    ]


def test_load_writes_csv_without_active_and_leaves_sources_alone(cache_dir):
    names = [BACE1_CSV_PATH.name, EGFR_CSV_PATH.name, "MolNet-tox21.csv"]
    before = {n: (cache_dir / n).read_bytes() for n in names}
    df = cl.load_combined(data_dir=cache_dir)

    written = pd.read_csv(cache_dir / cl.COMBINED_CSV_PATH.name)
    assert len(written) == len(df) == 4
    assert not any(c.startswith("active_") for c in written.columns)
    assert "active_BACE1" in df.columns
    assert {n: (cache_dir / n).read_bytes() for n in names} == before


def test_load_rebuilds_from_sources_every_time(cache_dir):
    assert len(cl.load_combined(data_dir=cache_dir)) == 4
    bio(["CCO"], [7.0]).to_csv(cache_dir / BACE1_CSV_PATH.name, index=False)
    df = cl.load_combined(data_dir=cache_dir)
    assert df["in_bace1"].sum() == 1
    assert "CCN" not in df["smiles"].tolist()


def test_refresh_reaches_every_loader(monkeypatch, tmp_path):
    calls = {}

    def fake(name, frame):
        def load(*args, **kwargs):
            calls[name] = kwargs["refresh"]
            return frame

        return load

    monkeypatch.setattr(cl, "load_bace1", fake("bace1", bio(["CCO"], [7.0])))
    monkeypatch.setattr(cl, "load_egfr", fake("egfr", bio(["CCO"], [7.0])))
    monkeypatch.setattr(
        cl,
        "load_toxicity_data",
        lambda name, refresh, data_dir: fake(name, tox(["CCO"], [[1, 0]]))(
            refresh=refresh
        ),
    )
    monkeypatch.setitem(DATASETS, "tox21", {"tasks": TASKS, "keep": "first"})
    monkeypatch.setitem(DATASETS, "clintox", {"tasks": TASKS, "keep": False})
    cl.load_combined(refresh=True, data_dir=tmp_path)
    assert calls == {"bace1": True, "egfr": True, "tox21": True, "clintox": True}


def test_overlap_counts_are_symmetric_with_source_sizes_on_the_diagonal(sources):
    overlap = cl.overlap_counts(cl.combine_sources(sources))
    assert (overlap.values == overlap.values.T).all()
    assert np.diag(overlap).tolist() == [2, 2, 2, 1]
    assert overlap.loc["in_bace1", "in_egfr"] == 1
    assert overlap.loc["in_tox21", "in_clintox"] == 1
