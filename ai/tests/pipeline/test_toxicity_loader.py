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
        return raw_dataset(["CCO", "c1ccccc1", "not_a_smiles"], [[1, np.nan]] * 3)

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


def test_clean_dataset_keep_first_keeps_first_duplicate():
    raw = raw_dataset(["CCO", "OCC.Cl", "c1ccccc1"], [[1, 0], [0, 1], [0, 0]])
    cleaned = tl.clean_dataset(raw, TASKS, "first")
    assert cleaned["smiles"].tolist() == ["CCO", "c1ccccc1"]
    assert cleaned.loc[0, TASKS].tolist() == [1, 0]


def test_clean_dataset_remove_all_drops_every_duplicate():
    raw = raw_dataset(["CCO", "OCC.Cl", "c1ccccc1"])
    cleaned = tl.clean_dataset(raw, TASKS, False)
    assert cleaned["smiles"].tolist() == ["c1ccccc1"]


def test_clean_dataset_drops_unparseable_and_extra_columns():
    raw = raw_dataset(["CCO", "not_a_smiles"]).assign(mol_id=["TOX1", "TOX2"])
    cleaned = tl.clean_dataset(raw, TASKS, "first")
    assert cleaned["smiles"].tolist() == ["CCO"]
    assert list(cleaned.columns) == ["smiles", *TASKS]


def test_clean_dataset_keeps_missing_labels_as_nan():
    cleaned = tl.clean_dataset(raw_dataset(["CCO"], [[1, np.nan]]), TASKS, "first")
    assert cleaned.loc[0, "T1"] == 1
    assert np.isnan(cleaned.loc[0, "T2"])


def test_unknown_dataset_raises_before_downloading(fake_fetch, tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        tl.load_toxicity_data("tox12", data_dir=tmp_path)
    assert fake_fetch == []


def test_downloads_then_uses_cache(fake_fetch, tmp_path):
    df = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert df["smiles"].tolist() == ["CCO", "c1ccccc1"]
    assert df["T1"].tolist() == [1, 1]
    assert df["T2"].isna().all()

    cached = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert fake_fetch == ["tox21"]
    pd.testing.assert_frame_equal(cached, df)
    assert [p.name for p in tmp_path.iterdir()] == ["MolNet-tox21.csv"]


def test_refresh_redownloads(fake_fetch, tmp_path):
    tl.load_toxicity_data("clintox", data_dir=tmp_path)
    tl.load_toxicity_data("clintox", refresh=True, data_dir=tmp_path)
    assert fake_fetch == ["clintox", "clintox"]


def test_stale_cache_columns_raise(fake_fetch, tmp_path):
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(tmp_path / "MolNet-tox21.csv", index=False)
    with pytest.raises(ValueError, match="--refresh"):
        tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert fake_fetch == []


def test_failed_write_leaves_no_cache(fake_fetch, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


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
