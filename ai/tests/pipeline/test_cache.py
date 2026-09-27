import pandas as pd
import pytest

from pipeline.cache import load_csv, write_csv


def fake_download(df):
    """A download callback that writes `df` and records each call."""
    calls = []

    def download(csv_path):
        calls.append(csv_path)
        write_csv(df, csv_path)

    return download, calls


def test_write_csv_creates_dirs_and_leaves_no_tmp(tmp_path):
    csv_path = tmp_path / "nested" / "data.csv"
    write_csv(pd.DataFrame({"smiles": ["CCO"]}), csv_path)
    assert pd.read_csv(csv_path)["smiles"].tolist() == ["CCO"]
    assert [p.name for p in csv_path.parent.iterdir()] == ["data.csv"]


def test_write_csv_failure_leaves_no_csv(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        write_csv(pd.DataFrame({"smiles": ["CCO"]}), tmp_path / "data.csv")
    assert list(tmp_path.iterdir()) == []


def test_load_csv_downloads_once_then_uses_cache(tmp_path):
    download, calls = fake_download(pd.DataFrame({"smiles": ["CCO"], "y": [1]}))
    csv_path = tmp_path / "data.csv"

    first = load_csv(csv_path, download, ["smiles", "y"])
    second = load_csv(csv_path, download, ["smiles", "y"])
    assert calls == [csv_path]
    pd.testing.assert_frame_equal(first, second)


def test_load_csv_refresh_redownloads(tmp_path):
    download, calls = fake_download(pd.DataFrame({"smiles": ["CCO"]}))
    csv_path = tmp_path / "data.csv"
    load_csv(csv_path, download, ["smiles"])
    load_csv(csv_path, download, ["smiles"], refresh=True)
    assert calls == [csv_path, csv_path]


def test_load_csv_missing_columns_says_to_refresh(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(csv_path, index=False)
    download, calls = fake_download(pd.DataFrame())
    with pytest.raises(ValueError, match=r"missing columns \['y'\].*--refresh"):
        load_csv(csv_path, download, ["smiles", "y"])
    assert calls == []
