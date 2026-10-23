import random

import numpy as np
import pandas as pd
import pytest
import torch

from baseline import benchmark, results


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "git_commit", lambda: "abc1234")
    data = tmp_path / "combined.csv"
    data.write_text("smiles\nCCO\n")
    return tmp_path / "results.csv", data


@pytest.fixture
def split_df():
    return pd.DataFrame(
        {
            "smiles": ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
            "split": ["val", "val", "val", "test", "test", "test"],
            "pIC50_BACE1": [5.0, 7.0, np.nan, 4.0, 8.0, np.nan],
            "active_BACE1": [0.0, 1.0, np.nan, 0.0, 1.0, np.nan],
            "pIC50_EGFR": [np.nan, 5.0, 7.0, np.nan, 5.0, 7.0],
            "active_EGFR": [np.nan, 0.0, 1.0, np.nan, 0.0, 1.0],
            **{task: [np.nan] * 6 for task in benchmark.TOX_TASKS},
        }
    ).assign(
        **{
            "NR-AR": [0.0, 1.0, np.nan, 0.0, 1.0, np.nan],
            "CT_TOX": [np.nan, 0.0, 1.0, np.nan, 0.0, 1.0],
        }
    )


def test_seed_everything_makes_random_numpy_and_torch_draws_repeat():
    gen_first = benchmark.seed_everything(123)
    py_first = [random.random() for _ in range(3)]
    np_first = np.random.rand(3)
    torch_first = torch.rand(3)
    perm_first = torch.randperm(20, generator=gen_first)

    gen_second = benchmark.seed_everything(123)
    py_second = [random.random() for _ in range(3)]
    np_second = np.random.rand(3)
    torch_second = torch.rand(3)
    perm_second = torch.randperm(20, generator=gen_second)

    assert py_first == py_second
    np.testing.assert_array_equal(np_first, np_second)
    assert torch.equal(torch_first, torch_second)
    assert torch.equal(perm_first, perm_second)


def test_evaluate_bioactivity_with_and_without_explicit_active_scores(split_df):
    # BACE1 has pIC50 only; EGFR provides both pIC50 and inverted active_EGFR on val
    predictions = {
        "val": {
            "pIC50_BACE1": [5.0, 7.0, np.nan],
            "pIC50_EGFR": [np.nan, 5.0, 7.0],
            "active_EGFR": [np.nan, 0.9, 0.1],
        },
        "test": {
            "pIC50_BACE1": [4.0, 8.0, np.nan],
            "active_EGFR": [np.nan, 0.1, 0.9],
        },
    }

    rows = pd.DataFrame(
        benchmark.evaluate_split_predictions(split_df, predictions)
    ).set_index(["endpoint", "split", "metric"])

    assert rows.loc[("pIC50_BACE1", "val", "rmse"), "value"] == pytest.approx(0.0)
    assert rows.loc[("pIC50_BACE1", "val", "r2"), "value"] == pytest.approx(1.0)
    assert rows.loc[("pIC50_BACE1", "val", "roc_auc"), "value"] == pytest.approx(1.0)
    assert rows.loc[("pIC50_BACE1", "val", "rmse"), "n_labeled"] == 2
    assert rows.loc[("pIC50_BACE1", "val", "rmse"), "n_positive"] == 1

    # pIC50-derived ROC-AUC under pIC50_EGFR, classifier head ROC-AUC under active_EGFR
    assert rows.loc[("pIC50_EGFR", "val", "rmse"), "value"] == pytest.approx(0.0)
    assert rows.loc[("pIC50_EGFR", "val", "roc_auc"), "value"] == pytest.approx(1.0)
    assert rows.loc[("active_EGFR", "val", "roc_auc"), "value"] == pytest.approx(0.0)
    assert ("active_EGFR", "val", "rmse") not in rows.index

    # Active-only predictions produce roc_auc under active_EGFR without pIC50_EGFR rows
    assert ("active_EGFR", "test", "roc_auc") in rows.index
    assert ("pIC50_EGFR", "test", "roc_auc") not in rows.index
    assert ("pIC50_EGFR", "test", "rmse") not in rows.index
    assert rows.loc[("active_EGFR", "test", "roc_auc"), "value"] == pytest.approx(1.0)


def test_evaluate_toxicity_excludes_omitted_endpoints_from_macros(split_df):
    predictions = {
        "val": {
            "NR-AR": [0.1, 0.9, np.nan],
        },
        "test": {
            "NR-AR": [0.1, 0.9, np.nan],
            "CT_TOX": [np.nan, 0.9, 0.1],
        },
    }

    rows = pd.DataFrame(
        benchmark.evaluate_split_predictions(split_df, predictions)
    ).set_index(["endpoint", "split"])

    # In val, only NR-AR (tox21) was predicted; CT_TOX (clintox) was omitted
    assert ("NR-AR", "val") in rows.index
    assert ("CT_TOX", "val") not in rows.index
    assert rows.loc[("NR-AR", "val"), "dataset"] == "tox21"
    assert rows.loc[("NR-AR", "val"), "value"] == pytest.approx(1.0)
    assert rows.loc[("NR-AR", "val"), "n_labeled"] == 2
    assert rows.loc[("NR-AR", "val"), "n_positive"] == 1

    assert rows.loc[("macro_tox21", "val"), "value"] == pytest.approx(1.0)
    assert np.isnan(rows.loc[("macro_clintox", "val"), "value"])
    assert rows.loc[("macro_overall", "val"), "value"] == pytest.approx(1.0)
    assert np.isnan(rows.loc[("macro_overall", "val"), "n_labeled"])

    # In test, NR-AR has AUC 1.0 and CT_TOX has AUC 0.0 -> macro_overall is 0.5
    assert rows.loc[("CT_TOX", "test"), "dataset"] == "clintox"
    assert rows.loc[("CT_TOX", "test"), "value"] == pytest.approx(0.0)
    assert rows.loc[("macro_overall", "test"), "value"] == pytest.approx(0.5)


def test_evaluate_rejects_nan_on_labeled_row(split_df):
    predictions = {
        "val": {
            "pIC50_BACE1": [5.0, np.nan, np.nan],
        }
    }
    with pytest.raises(ValueError, match="NaN on labeled row"):
        benchmark.evaluate_split_predictions(split_df, predictions)


def test_evaluate_rejects_wrong_length_split_or_unknown_column(split_df):
    with pytest.raises(ValueError, match="unknown split"):
        benchmark.evaluate_split_predictions(split_df, {"holdout": {}})

    with pytest.raises(ValueError, match="unknown prediction column"):
        benchmark.evaluate_split_predictions(split_df, {"val": {"unknown": [1, 2, 3]}})

    with pytest.raises(ValueError, match="expected 1D array of length 3"):
        benchmark.evaluate_split_predictions(
            split_df, {"val": {"pIC50_BACE1": [5.0, 7.0]}}
        )


def test_run_benchmark_records_per_seed_and_returns_summary(split_df, paths):
    path, data_path = paths
    seen_seeds = []

    def fit_predict(df, seed):
        seen_seeds.append(seed)
        offset = 0.2 * seed
        return {
            "val": {"pIC50_BACE1": [5.0 + offset, 7.0 + offset, np.nan]},
            "test": {"pIC50_BACE1": [4.0 + offset, 8.0 + offset, np.nan]},
        }

    summary = benchmark.run_benchmark(
        fit_predict,
        model="stub",
        encoder="enc",
        params={"b": 2, "a": 1},
        seeds=(0, 1, 2),
        df=split_df,
        path=path,
        data_path=data_path,
    )

    assert seen_seeds == [0, 1, 2]
    written = pd.read_csv(path)
    # 2 splits * 3 metrics * 3 seeds = 18 rows
    assert len(written) == 18
    assert sorted(written["seed"].unique()) == [0, 1, 2]
    assert (written["params"] == '{"a": 1, "b": 2}').all()

    indexed = summary.set_index(["split", "metric"])
    assert (indexed["n_seeds"] == 3).all()
    # RMSE on [0.0, 0.2, 0.4] has mean 0.2 and sample std 0.2
    assert indexed.loc[("val", "rmse"), "value"] == pytest.approx(0.2)
    assert indexed.loc[("val", "rmse"), "std"] == pytest.approx(0.2)
    assert indexed.loc[("test", "roc_auc"), "value"] == pytest.approx(1.0)
    assert indexed.loc[("test", "roc_auc"), "std"] == pytest.approx(0.0)


def test_run_benchmark_rejects_empty_seeds(split_df, paths):
    path, data_path = paths
    with pytest.raises(ValueError, match="seeds must not be empty"):
        benchmark.run_benchmark(
            lambda df, seed: {},
            model="stub",
            encoder="enc",
            params={},
            seeds=(),
            df=split_df,
            path=path,
            data_path=data_path,
        )


def test_load_benchmark_split_delegates_to_combined_and_scaffold(monkeypatch):
    sentinel = pd.DataFrame({"smiles": ["CCO"]})
    monkeypatch.setattr(benchmark, "load_combined", lambda: sentinel)
    monkeypatch.setattr(
        benchmark, "scaffold_split", lambda df: df.assign(split=["train"])
    )
    out = benchmark.load_benchmark_split()
    assert out["split"].tolist() == ["train"]
