import gzip
import io
import sys
import urllib.request

import numpy as np
import pandas as pd
import pytest

from pipeline import toxicity_loader as tl

TASKS = ["T1", "T2"]


def raw_dataset(smiles, labels=None):
    labels = [[0, 0]] * len(smiles) if labels is None else labels
    return pd.DataFrame(labels, columns=TASKS).assign(smiles=smiles)


@pytest.fixture
def fake_fetch(monkeypatch):
    calls = []

    def fetch(name):
        calls.append(name)
        raw = raw_dataset(["CCO", "c1ccccc1", "not_a_smiles"], [[1, np.nan]] * 3)
        return raw.assign(mol_id=["TOX1", "TOX2", "TOX3"])

    monkeypatch.setattr(tl, "fetch_dataset", fetch)
    monkeypatch.setitem(tl.DATASETS, "tox21", {"tasks": TASKS, "keep": "first"})
    monkeypatch.setitem(tl.DATASETS, "clintox", {"tasks": TASKS, "keep": False})
    return calls


def gzipped_csv(df):
    return io.BytesIO(gzip.compress(df.to_csv(index=False).encode()))


def test_fetch_dataset_reads_gzipped_csv(monkeypatch):
    urls = []

    def fake_urlopen(url, timeout=None):
        urls.append(url)
        return gzipped_csv(raw_dataset(["CCO"], [[1, np.nan]]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(tl.DATASETS, "tox21", {"tasks": TASKS, "keep": "first"})
    raw = tl.fetch_dataset("tox21")

    assert urls == [tl.MOLECULENET_URL.format(name="tox21")]
    assert raw["smiles"].tolist() == ["CCO"]
    assert np.isnan(raw.loc[0, "T2"])


def test_fetch_dataset_rejects_missing_columns(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=None: gzipped_csv(pd.DataFrame({"smiles": ["CCO"]})),
    )
    monkeypatch.setitem(tl.DATASETS, "tox21", {"tasks": TASKS, "keep": "first"})
    with pytest.raises(RuntimeError, match="missing columns"):
        tl.fetch_dataset("tox21")


# OCC.Cl standardizes to CCO, duplicating the first row
@pytest.mark.parametrize(
    ("keep", "expected"),
    [("first", ["CCO", "c1ccccc1"]), (False, ["c1ccccc1"])],
)
def test_clean_dataset_duplicates(keep, expected):
    raw = raw_dataset(["CCO", "OCC.Cl", "c1ccccc1"], [[1, 0], [0, 1], [0, 0]])
    cleaned = tl.clean_dataset(raw, TASKS, keep)
    assert cleaned["smiles"].tolist() == expected
    if keep == "first":
        assert cleaned.loc[0, TASKS].tolist() == [1, 0]


def test_unknown_dataset_raises_before_downloading(fake_fetch, tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        tl.load_toxicity_data("tox12", data_dir=tmp_path)
    assert fake_fetch == []


def test_downloads_then_uses_cache(fake_fetch, tmp_path):
    df = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert list(df.columns) == ["smiles", *TASKS]
    assert df["smiles"].tolist() == ["CCO", "c1ccccc1"]
    assert df["T1"].tolist() == [1, 1]
    assert df["T2"].isna().all()

    cached = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert fake_fetch == ["tox21"]
    pd.testing.assert_frame_equal(cached, df)
    assert [p.name for p in tmp_path.iterdir()] == ["MolNet-tox21.csv"]


def test_main_forwards_flags_and_prints_a_summary(monkeypatch, capsys):
    seen = []

    def fake_load(name, refresh):
        seen.append((name, refresh))
        return raw_dataset(["CCO"], [[1, 0]])

    monkeypatch.setattr(tl, "load_toxicity_data", fake_load)
    monkeypatch.setitem(tl.DATASETS, "clintox", {"tasks": TASKS, "keep": False})
    monkeypatch.setattr(
        sys, "argv", ["pipeline.toxicity_loader", "--refresh", "--dataset", "clintox"]
    )
    tl.main()

    assert seen == [("clintox", True)]
    out = capsys.readouterr().out
    assert "clintox: 1 molecules" in out
    assert "positive_rate" in out


def test_main_defaults_to_all_datasets(monkeypatch):
    seen = []

    def fake_load(name, refresh):
        seen.append((name, refresh))
        return raw_dataset(["CCO"])

    monkeypatch.setattr(tl, "load_toxicity_data", fake_load)
    monkeypatch.setitem(tl.DATASETS, "tox21", {"tasks": TASKS, "keep": "first"})
    monkeypatch.setitem(tl.DATASETS, "clintox", {"tasks": TASKS, "keep": False})
    monkeypatch.setattr(sys, "argv", ["pipeline.toxicity_loader"])
    tl.main()
    assert seen == [("tox21", False), ("clintox", False)]


@pytest.mark.network
def test_real_moleculenet_download(tmp_path):
    for name, n_tasks in [("tox21", 12), ("clintox", 2)]:
        df = tl.load_toxicity_data(name, data_dir=tmp_path)
        assert list(df.columns) == ["smiles", *tl.DATASETS[name]["tasks"]]
        assert len(tl.DATASETS[name]["tasks"]) == n_tasks
        assert df["smiles"].is_unique
        assert "." not in "".join(df["smiles"])
        labels = df[tl.DATASETS[name]["tasks"]]
        assert (labels.isin([0, 1]) | labels.isna()).all().all()
