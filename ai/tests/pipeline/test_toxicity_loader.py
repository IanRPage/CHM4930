import sys

import numpy as np
import pytest
from deepchem.data import NumpyDataset

from pipeline import toxicity_loader as tl

TASKS = ["T1", "T2"]


def raw_dataset(smiles, y=None, w=None):
    """A stand-in for what deepchem's load_tox21/load_clintox return, unsplit."""
    n = len(smiles)
    y = np.zeros((n, len(TASKS))) if y is None else np.asarray(y, dtype=float)
    w = np.ones((n, len(TASKS))) if w is None else np.asarray(w, dtype=float)
    return NumpyDataset(X=np.array(smiles), y=y, w=w, ids=np.array(smiles))


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replaces the MoleculeNet download and records which datasets were fetched."""
    calls = []

    def fetch(name):
        calls.append(name)
        return TASKS, (raw_dataset(["CCO", "c1ccccc1", "not_a_smiles"]),), []

    monkeypatch.setattr(tl, "_fetch", fetch)
    return calls


def test_clean_dataset_keep_first_keeps_first_duplicate():
    raw = raw_dataset(["CCO", "OCC.Cl", "c1ccccc1"], y=[[1, 0], [0, 1], [0, 0]])
    cleaned = tl.clean_dataset(raw, "keep_first")
    assert list(cleaned.ids) == ["CCO", "c1ccccc1"]
    assert cleaned.y[0].tolist() == [1, 0]


def test_clean_dataset_remove_all_drops_every_duplicate():
    raw = raw_dataset(["CCO", "OCC.Cl", "c1ccccc1"])
    cleaned = tl.clean_dataset(raw, "remove_all")
    assert list(cleaned.ids) == ["c1ccccc1"]


def test_clean_dataset_drops_unparseable():
    cleaned = tl.clean_dataset(raw_dataset(["CCO", "not_a_smiles"]), "keep_first")
    assert list(cleaned.ids) == ["CCO"]


def test_unknown_dataset_raises_before_downloading(fake_fetch, tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        tl.load_toxicity_data("tox12", data_dir=tmp_path)
    assert fake_fetch == []


def test_downloads_then_uses_cache(fake_fetch, tmp_path):
    tasks, (dataset,), transformers = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert tasks == TASKS
    assert list(dataset.ids) == ["CCO", "c1ccccc1"]
    assert transformers == []

    tasks, (dataset,), _ = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert fake_fetch == ["tox21"]
    assert list(dataset.ids) == ["CCO", "c1ccccc1"]
    assert not (tmp_path / "MolNet-tox21.tmp").exists()


def test_refresh_redownloads(fake_fetch, tmp_path):
    tl.load_toxicity_data("clintox", data_dir=tmp_path)
    tl.load_toxicity_data("clintox", refresh=True, data_dir=tmp_path)
    assert fake_fetch == ["clintox", "clintox"]


def test_incomplete_cache_is_rebuilt(fake_fetch, tmp_path):
    tl.load_toxicity_data("tox21", data_dir=tmp_path)
    (tmp_path / "MolNet-tox21" / "tasks.json").unlink()

    _, (dataset,), _ = tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert fake_fetch == ["tox21", "tox21"]
    assert len(dataset) == 2


def test_failed_write_leaves_no_cache(fake_fetch, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tl, "save_to_disk", boom)
    with pytest.raises(OSError):
        tl.load_toxicity_data("tox21", data_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_summarize_counts_missing_and_positives():
    dataset = raw_dataset(
        ["C", "CC", "CCC"],
        y=[[1, 0], [0, 0], [1, 0]],
        w=[[1, 0], [1, 0], [1, 0]],
    )
    summary = tl.summarize(TASKS, dataset)
    assert summary.loc["T1"].tolist() == pytest.approx([3, 0, 2 / 3])
    assert summary.loc["T2", "labeled"] == 0
    assert summary.loc["T2", "missing"] == 3
    assert np.isnan(summary.loc["T2", "positive_rate"])


def test_main_forwards_flags_and_prints_a_summary(monkeypatch, capsys):
    seen = []

    def fake_load(name, refresh):
        seen.append((name, refresh))
        return TASKS, (raw_dataset(["CCO"], y=[[1, 0]]),), []

    monkeypatch.setattr(tl, "load_toxicity_data", fake_load)
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
        return TASKS, (raw_dataset(["CCO"]),), []

    monkeypatch.setattr(tl, "load_toxicity_data", fake_load)
    monkeypatch.setattr(sys, "argv", ["pipeline.toxicity_loader"])
    tl.main()
    assert seen == [("tox21", False), ("clintox", False)]


@pytest.mark.network
def test_real_moleculenet_download(tmp_path):
    for name, n_tasks in [("tox21", 12), ("clintox", 2)]:
        tasks, (dataset,), _ = tl.load_toxicity_data(name, data_dir=tmp_path)
        assert len(tasks) == n_tasks
        assert dataset.y.shape == (len(dataset), n_tasks)
        assert len(set(dataset.ids)) == len(dataset)
        assert "." not in "".join(dataset.ids)  # no multi-fragment structures left
