import hashlib
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


def row(endpoint, split="test", seed=0, value=0.5, **extra):
    return {
        "encoder": "ecfp4-2048",
        "dataset": "tox21",
        "endpoint": endpoint,
        "split": split,
        "seed": seed,
        "metric": "roc_auc",
        "value": value,
        "n_labeled": 10,
        "n_positive": 3,
        "params": "{}",
        **extra,
    }


def test_run_info_is_stored_once_per_run_and_joined_on_load(paths):
    path, data = paths
    rows = [row("T1"), row("T2"), row("T1", seed=1)]
    recorded = results.record_results(rows, model="m", path=path, data_path=data)

    assert pd.read_csv(path).columns.tolist() == results.METRIC_COLUMNS
    runs = pd.read_csv(results.runs_path(path))
    assert runs.columns.tolist() == results.RUN_COLUMNS
    assert runs["seed"].tolist() == [0, 1]
    written = results.load_results(path)
    assert written.columns.tolist() == results.COLUMNS
    pd.testing.assert_frame_equal(written, recorded)
    assert written.loc[0, "model"] == "m"
    assert written.loc[0, "seed"] == 0
    assert written.loc[0, "git_commit"] == "abc1234"


def test_failed_write_leaves_no_partial_files(paths, monkeypatch):
    path, data = paths
    original = type(path).replace

    def fail_on_runs(self, target):
        if target.name == results.runs_path(path).name:
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(type(path), "replace", fail_on_runs)

    with pytest.raises(OSError, match="disk full"):
        results.record_results([row("T1")], model="m", path=path, data_path=data)
    assert not list(path.parent.glob("*.tmp"))


def test_load_results_on_missing_file_is_empty(tmp_path):
    loaded = results.load_results(tmp_path / "missing.csv")

    assert loaded.empty
    assert loaded.columns.tolist() == results.COLUMNS


def test_run_with_inconsistent_run_info_is_rejected(paths):
    path, data = paths
    rows = [row("T1"), row("T2", params='{"lr": 1}')]

    with pytest.raises(ValueError, match="more than one"):
        results.record_results(rows, model="m", path=path, data_path=data)
    assert not path.exists()
    assert not results.runs_path(path).exists()


def test_params_json_is_key_order_independent():
    first = results.params_json(
        {"n_estimators": 500, "max_features": {"EGFR": "sqrt", "BACE1": 0.1}}
    )
    second = results.params_json(
        {"max_features": {"BACE1": 0.1, "EGFR": "sqrt"}, "n_estimators": 500}
    )
    assert first == second


def test_rerun_replaces_only_that_models_rows(paths):
    path, data = paths
    results.record_results([row("T1"), row("T2")], model="a", path=path, data_path=data)
    results.record_results([row("T1")], model="b", path=path, data_path=data)
    results.record_results([row("T1", value=0.9)], model="a", path=path, data_path=data)
    written = results.load_results(path)
    assert len(written) == 2
    assert written.set_index("model").loc["a", "value"] == 0.9
    assert written["model"].tolist() == ["a", "b"]


def test_other_models_stale_rows_are_dropped_only_with_opt_in(paths, caplog):
    path, data = paths
    results.record_results([row("T1")], model="a", path=path, data_path=data)
    results.record_results([row("T1")], model="b", path=path, data_path=data)
    before = path.read_bytes(), results.runs_path(path).read_bytes()
    data.write_text("smiles\nCCN\n")

    with pytest.raises(ValueError, match="different snapshot"):
        results.record_results([row("T1")], model="a", path=path, data_path=data)
    assert (path.read_bytes(), results.runs_path(path).read_bytes()) == before

    results.record_results(
        [row("T1")], model="a", path=path, data_path=data, allow_drop_stale=True
    )
    written = results.load_results(path)
    assert written["model"].tolist() == ["a"]
    assert pd.read_csv(results.runs_path(path))["model"].tolist() == ["a"]
    assert written["data_sha256"].tolist() == [results.file_sha256(data)]
    assert "['b']" in caplog.text


def test_rerun_replaces_only_matching_encoder_and_seed_rows(paths):
    path, data = paths
    rows = [row("T1", seed=s, encoder=e) for e in ("gnn", "ecfp") for s in (0, 1)]
    results.record_results(rows, model="m", path=path, data_path=data)
    results.record_results(
        [row("T1", seed=0, value=0.9, encoder="ecfp")],
        model="m",
        path=path,
        data_path=data,
    )

    written = results.load_results(path).set_index(["encoder", "seed"])["value"]
    assert written.to_dict() == {
        ("ecfp", 0): 0.9,
        ("ecfp", 1): 0.5,
        ("gnn", 0): 0.5,
        ("gnn", 1): 0.5,
    }


def test_rows_are_sorted_by_seed_and_nan_values_kept(paths):
    path, data = paths
    rows = [
        row("T2", "val", seed=0),
        row("T1", "val", seed=0, value=np.nan),
        row("T1", "test", seed=1),
        row("T1", "test", seed=0),
    ]
    results.record_results(rows, model="m", path=path, data_path=data)
    written = results.load_results(path)
    assert written[["endpoint", "split", "seed"]].values.tolist() == [
        ["T1", "test", 0],
        ["T1", "test", 1],
        ["T1", "val", 0],
        ["T2", "val", 0],
    ]
    assert np.isnan(written.loc[2, "value"])


def test_summarize_seeds_computes_mean_ddof1_std_and_keeps_nan_keys(paths):
    path, data = paths
    rows = [
        row("T1", "test", seed=0, value=0.6),
        row("T1", "test", seed=1, value=0.8),
        row("T1", "test", seed=2, value=1.0),
        row("T1", "test", seed=3, value=np.nan),
        row(
            "macro_tox21",
            "test",
            seed=0,
            value=0.5,
            n_labeled=np.nan,
            n_positive=np.nan,
        ),
        row(
            "macro_tox21",
            "test",
            seed=1,
            value=0.9,
            n_labeled=np.nan,
            n_positive=np.nan,
        ),
    ]
    written = results.record_results(rows, model="m", path=path, data_path=data)
    summary = results.summarize_seeds(written).set_index("endpoint")

    assert list(summary.columns) == [
        c for c in results.SUMMARY_COLUMNS if c != "endpoint"
    ]
    assert summary.loc["T1", "value"] == pytest.approx(0.8)
    assert summary.loc["T1", "std"] == pytest.approx(0.2)
    assert summary.loc["T1", "n_seeds"] == 3

    assert summary.loc["macro_tox21", "value"] == pytest.approx(0.7)
    assert summary.loc["macro_tox21", "std"] == pytest.approx(
        np.std([0.5, 0.9], ddof=1)
    )
    assert summary.loc["macro_tox21", "n_seeds"] == 2
    assert np.isnan(summary.loc["macro_tox21", "n_labeled"])
    assert np.isnan(summary.loc["macro_tox21", "n_positive"])


def test_summarize_seeds_empty_dataframe():
    empty = pd.DataFrame(columns=results.COLUMNS)
    summary = results.summarize_seeds(empty)
    assert list(summary.columns) == results.SUMMARY_COLUMNS
    assert summary.empty


def test_data_sha256_tracks_the_snapshot_bytes(paths):
    path, data = paths
    first = results.file_sha256(data)
    assert first == results.file_sha256(data)
    results.record_results([row("T1")], model="m", path=path, data_path=data)
    assert results.load_results(path).loc[0, "data_sha256"] == first
    data.write_text("smiles\nCCN\n")
    assert results.file_sha256(data) != first


def test_lf_and_crlf_checksums_match_without_rewriting_files(tmp_path):
    lf = b"smiles,pIC50_BACE1\nCCO,6.0\nCCN,5.0\n"
    expected = hashlib.sha256(lf).hexdigest()
    for name, contents in (("lf.csv", lf), ("crlf.csv", lf.replace(b"\n", b"\r\n"))):
        path = tmp_path / name
        path.write_bytes(contents)
        assert results.file_sha256(path) == expected
        assert path.read_bytes() == contents


def test_git_commit_falls_back_when_git_is_unavailable(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert results.git_commit() == "unknown"


def test_git_commit_marks_a_dirty_tree_ignoring_results_csv(monkeypatch):
    outputs = {"rev-parse": "abc1234", "status": " M file.py"}
    status_args = []

    def run(args, **kwargs):
        if args[1] == "status":
            status_args.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[1]] + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert results.git_commit() == "abc1234-dirty"
    outputs["status"] = ""
    assert results.git_commit() == "abc1234"
    for excluded in (results.RESULTS_PATH, results.runs_path()):
        assert all(f":(exclude){excluded}" in a for a in status_args)
