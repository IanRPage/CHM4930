import subprocess

import numpy as np
import pandas as pd
import pytest

from baseline import results


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "git_commit", lambda: "abc1234")
    data = tmp_path / "combined.csv"
    data.write_text("smiles\nCCO\n")
    return tmp_path / "baseline" / "results.csv", data


def row(endpoint, split="test", value=0.5, **extra):
    return {
        "encoder": "ecfp4-2048",
        "dataset": "tox21",
        "endpoint": endpoint,
        "split": split,
        "metric": "roc_auc",
        "value": value,
        "n_labeled": 10,
        "n_positive": 3,
        "params": "{}",
        **extra,
    }


def test_rows_get_every_column_in_order(paths):
    path, data = paths
    results.record_results([row("T1")], model="m", path=path, data_path=data)
    written = pd.read_csv(path)
    assert written.columns.tolist() == results.COLUMNS
    assert written.loc[0, "model"] == "m"
    assert written.loc[0, "git_commit"] == "abc1234"


def test_rerun_replaces_only_that_models_rows(paths):
    path, data = paths
    results.record_results([row("T1"), row("T2")], model="a", path=path, data_path=data)
    results.record_results([row("T1")], model="b", path=path, data_path=data)
    results.record_results([row("T1", value=0.9)], model="a", path=path, data_path=data)
    written = pd.read_csv(path)
    assert len(written) == 2
    assert written.set_index("model").loc["a", "value"] == 0.9
    assert written["model"].tolist() == ["a", "b"]


def test_other_models_rows_from_an_old_snapshot_are_dropped(paths, caplog):
    path, data = paths
    results.record_results([row("T1")], model="a", path=path, data_path=data)
    results.record_results([row("T1")], model="b", path=path, data_path=data)
    data.write_text("smiles\nCCN\n")
    results.record_results([row("T1")], model="a", path=path, data_path=data)

    written = pd.read_csv(path)
    assert written["model"].tolist() == ["a"]
    assert written["data_sha256"].tolist() == [results.file_sha256(data)]
    assert "['b']" in caplog.text


def test_rows_are_sorted_and_nan_values_kept(paths):
    path, data = paths
    rows = [row("T2", "val"), row("T1", "val", value=np.nan), row("T1", "test")]
    results.record_results(rows, model="m", path=path, data_path=data)
    written = pd.read_csv(path)
    assert written[["endpoint", "split"]].values.tolist() == [
        ["T1", "test"],
        ["T1", "val"],
        ["T2", "val"],
    ]
    assert np.isnan(written.loc[1, "value"])


def test_data_sha256_tracks_the_snapshot_bytes(paths):
    path, data = paths
    first = results.file_sha256(data)
    assert first == results.file_sha256(data)
    results.record_results([row("T1")], model="m", path=path, data_path=data)
    assert pd.read_csv(path).loc[0, "data_sha256"] == first
    data.write_text("smiles\nCCN\n")
    assert results.file_sha256(data) != first


def test_git_commit_falls_back_when_git_is_unavailable(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert results.git_commit() == "unknown"


def test_git_commit_marks_a_dirty_tree(monkeypatch):
    outputs = {"rev-parse": "abc1234", "status": " M file.py"}

    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[1]] + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert results.git_commit() == "abc1234-dirty"
    outputs["status"] = ""
    assert results.git_commit() == "abc1234"
