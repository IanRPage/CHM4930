import hashlib
import subprocess

import numpy as np
import pandas as pd
import pytest

from baseline import results


def row(endpoint, split="test", seed=0, value=0.5, **extra):
    return {
        "model": "m",
        "encoder": "ecfp4-2048",
        "dataset": "tox21",
        "endpoint": endpoint,
        "split_id": "default",
        "split": split,
        "seed": seed,
        "metric": "roc_auc",
        "value": value,
        "n_labeled": 10,
        "n_positive": 3,
        "run_id": "m_default",
        "n_seeds": 2,
        "params": "{}",
        "data_sha256": "d" * 64,
        "split_sha256": "s" * 64,
        "base_commit": "a" * 40,
        "ai_tree": "b" * 40,
        "patch_sha256": "",
        "date": "2026-10-30",
        **extra,
    }


def test_run_info_is_stored_once_per_run_and_joined_on_load(tmp_path):
    path = tmp_path / "baseline" / "results.csv"
    df = pd.DataFrame([row("T1"), row("T2"), row("T1", seed=1)])
    results.write_results(df, path)

    assert pd.read_csv(path).columns.tolist() == results.METRIC_COLUMNS
    runs = pd.read_csv(results.runs_path(path))
    assert runs.columns.tolist() == results.RUN_COLUMNS
    assert len(runs) == 1
    assert runs.loc[0, "run_id"] == "m_default"
    assert runs.loc[0, "n_seeds"] == 2
    written = results.load_results(path)
    assert written.columns.tolist() == results.COLUMNS
    assert len(written) == 3
    assert written.loc[0, "base_commit"] == "a" * 40
    assert written.loc[0, "patch_sha256"] == ""


@pytest.mark.parametrize("existing", [False, True])
def test_failed_replace_restores_both_files(tmp_path, monkeypatch, existing):
    path = tmp_path / "baseline" / "results.csv"
    runs = results.runs_path(path)
    if existing:
        results.write_results(pd.DataFrame([row("T1", model="a")]), path)
    before = {p: p.read_bytes() for p in (path, runs) if p.exists()}
    original = type(path).replace
    failed = []

    def fail_once_on_runs(self, target):
        if target.name == runs.name and not failed:
            failed.append(target)
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(type(path), "replace", fail_once_on_runs)

    with pytest.raises(OSError, match="disk full"):
        results.write_results(pd.DataFrame([row("T1", model="b")]), path)
    assert {p: p.read_bytes() for p in (path, runs) if p.exists()} == before
    assert not list(path.parent.glob("*.tmp")) + list(path.parent.glob("*.bak"))


def test_load_results_on_missing_file_is_empty(tmp_path):
    loaded = results.load_results(tmp_path / "missing.csv")
    assert loaded.empty
    assert loaded.columns.tolist() == results.COLUMNS


def test_run_with_inconsistent_run_info_is_rejected(tmp_path):
    path = tmp_path / "results.csv"
    df = pd.DataFrame([row("T1"), row("T2", params='{"lr": 1}')])

    with pytest.raises(ValueError, match="more than one"):
        results.write_results(df, path)
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


def test_rows_are_sorted_by_seed_and_nan_values_kept(tmp_path):
    path = tmp_path / "results.csv"
    df = pd.DataFrame(
        [
            row("T2", "val", seed=0),
            row("T1", "val", seed=0, value=np.nan),
            row("T1", "test", seed=1),
            row("T1", "test", seed=0),
        ]
    )
    results.write_results(df, path)
    written = results.load_results(path)
    assert written[["endpoint", "split", "seed"]].values.tolist() == [
        ["T1", "test", 0],
        ["T1", "test", 1],
        ["T1", "val", 0],
        ["T2", "val", 0],
    ]
    assert np.isnan(written.loc[2, "value"])


def test_summarize_seeds_computes_mean_ddof1_std_and_keeps_nan_keys():
    df = pd.DataFrame(
        [
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
    )
    summary = results.summarize_seeds(df).set_index("endpoint")

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


def test_lf_and_crlf_checksums_match_without_rewriting_files(tmp_path):
    lf = b"smiles,pIC50_BACE1\nCCO,6.0\nCCN,5.0\n"
    expected = hashlib.sha256(lf).hexdigest()
    for name, contents in (("lf.csv", lf), ("crlf.csv", lf.replace(b"\n", b"\r\n"))):
        path = tmp_path / name
        path.write_bytes(contents)
        assert results.file_sha256(path) == expected
        assert path.read_bytes() == contents


def test_code_identity_scopes_to_code_subtrees_and_verifies_patches(tmp_path):
    repo = tmp_path / "repo"
    ai_dir = repo / "ai"
    (ai_dir / "src").mkdir(parents=True)
    (ai_dir / "drivers").mkdir(parents=True)
    (ai_dir / "data" / "baseline").mkdir(parents=True)
    (ai_dir / "src" / "mod.py").write_text("x = 1\n")
    (ai_dir / "drivers" / "run.py").write_text("print(1)\n")
    (ai_dir / "environment.yml").write_text("name: test\n")
    (ai_dir / "README.md").write_text("# Doc\n")

    def git(*args):
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-m", "initial")

    clean = results.code_identity(ai_dir)
    assert len(clean.base_commit) == 40
    assert len(clean.ai_tree) == 40
    assert clean.patch == ""
    assert clean.patch_sha256 == ""
    results.verify_code_identity(
        clean.base_commit, clean.ai_tree, clean.patch_sha256, ai_dir=ai_dir
    )

    # Editing docs or data/baseline does NOT change ai_tree or produce a patch
    (ai_dir / "README.md").write_text("# Updated doc\n")
    (ai_dir / "data" / "baseline" / "results.csv").write_text("a,b\n1,2\n")
    after_doc = results.code_identity(ai_dir)
    assert after_doc.ai_tree == clean.ai_tree
    assert after_doc.patch == ""

    # Editing src/ changes ai_tree and produces a reproducible patch
    (ai_dir / "src" / "mod.py").write_text("x = 2\n")
    dirty = results.code_identity(ai_dir)
    assert dirty.base_commit == clean.base_commit
    assert dirty.ai_tree != clean.ai_tree
    assert dirty.patch_sha256 != ""
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text(dirty.patch)
    results.verify_code_identity(
        dirty.base_commit,
        dirty.ai_tree,
        dirty.patch_sha256,
        patch_path=patch_file,
        ai_dir=ai_dir,
    )


def summary_rows(model, split_id, values, metric="rmse"):
    return [
        {
            "model": model,
            "encoder": "enc",
            "dataset": "bace1",
            "endpoint": "pIC50_BACE1",
            "split_id": split_id,
            "split": "test",
            "seed": seed,
            "metric": metric,
            "value": value,
            "ci_low": value - 0.1,
            "ci_high": value + 0.1,
            "data_sha256": "sha",
        }
        for seed, value in enumerate(values)
    ]


def test_split_and_seed_variance_are_reported_separately():
    df = pd.DataFrame(
        summary_rows("m", "default", [1.0, 1.2])
        + summary_rows("m", "scaffold-0", [2.0, 2.2])
    ).reindex(columns=results.COLUMNS)

    per_split = results.summarize_seeds(df).set_index("split_id")
    assert per_split.loc["default", "value"] == pytest.approx(1.1)
    assert per_split.loc["default", "ci_low"] == pytest.approx(1.0)
    assert per_split.loc["default", "std"] == pytest.approx(np.sqrt(0.02))

    (row_dict,) = results.summarize_splits(df).to_dict("records")
    assert row_dict["value"] == pytest.approx(1.6)
    assert row_dict["split_std"] == pytest.approx(np.sqrt(0.5))
    assert row_dict["seed_std"] == pytest.approx(np.sqrt(0.02))
    assert (row_dict["n_splits"], row_dict["n_seeds"]) == (2, 4)
