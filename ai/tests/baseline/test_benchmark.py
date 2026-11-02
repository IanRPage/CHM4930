import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from baseline import benchmark, results
from pipeline.bioactivity_loader import PIC50_ACTIVE_THRESHOLD
from pipeline.combined_loader import COMBINED_CSV_PATH
from pipeline.split import scaffold_split


@pytest.fixture
def paths(tmp_path):
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
    assert rows.loc[("pIC50_BACE1", "val", "spearman"), "value"] == pytest.approx(1.0)
    assert rows.loc[("pIC50_BACE1", "val", "pr_auc"), "value"] == pytest.approx(1.0)
    assert rows.loc[("pIC50_BACE1", "val", "prevalence"), "value"] == 0.5
    assert rows.loc[("pIC50_BACE1", "val", "rmse"), "n_labeled"] == 2
    assert rows.loc[("pIC50_BACE1", "val", "rmse"), "n_positive"] == 1
    assert rows["ci_low"].isna().all()

    assert rows.loc[("pIC50_EGFR", "val", "rmse"), "value"] == pytest.approx(0.0)
    assert rows.loc[("pIC50_EGFR", "val", "roc_auc"), "value"] == pytest.approx(1.0)
    assert rows.loc[("active_EGFR", "val", "roc_auc"), "value"] == pytest.approx(0.0)
    assert ("active_EGFR", "val", "rmse") not in rows.index

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

    rows = pd.DataFrame(benchmark.evaluate_split_predictions(split_df, predictions))
    rows = rows[rows["metric"] == "roc_auc"].set_index(["endpoint", "split"])

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
    assert rows.loc[("macro_overall", "val"), "n_tasks"] == 1
    dropped = rows.loc[("macro_overall", "val"), "dropped_tasks"].split(";")
    assert "CT_TOX" in dropped and "NR-AR" not in dropped
    assert len(dropped) == len(benchmark.TOX_TASKS) - 1

    assert rows.loc[("CT_TOX", "test"), "dataset"] == "clintox"
    assert rows.loc[("CT_TOX", "test"), "value"] == pytest.approx(0.0)
    assert rows.loc[("macro_overall", "test"), "value"] == pytest.approx(0.5)


def test_evaluate_checks_optional_smiles_alignment(split_df):
    rows = benchmark.evaluate_split_predictions(
        split_df,
        {
            "val": {
                "smiles": ["C", "CC", "CCC"],
                "pIC50_BACE1": [5.0, 7.0, np.nan],
            }
        },
    )
    assert len(rows) == 7

    with pytest.raises(ValueError, match="prediction SMILES do not match"):
        benchmark.evaluate_split_predictions(
            split_df,
            {
                "val": {
                    "smiles": ["CC", "C", "CCC"],
                    "pIC50_BACE1": [5.0, 7.0, np.nan],
                }
            },
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_evaluate_rejects_nonfinite_score_on_labeled_row(split_df, bad):
    predictions = {
        "val": {
            "pIC50_BACE1": [5.0, bad, np.inf],
        }
    }
    with pytest.raises(ValueError, match="non-finite score on labeled row"):
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


def test_bootstrap_ci_is_test_only_brackets_value_and_repeats_with_seed():
    rng = np.random.default_rng(0)
    n = 60
    y = rng.normal(6.0, 1.0, n)
    df = pd.DataFrame(
        {
            "smiles": [f"C{i}" for i in range(n)],
            "split": ["val"] * 30 + ["test"] * 30,
            "pIC50_BACE1": y,
            "active_BACE1": (y >= 6).astype(float),
            "pIC50_EGFR": np.nan,
            "active_EGFR": np.nan,
            **{task: np.nan for task in benchmark.TOX_TASKS},
        }
    )
    noisy = y + rng.normal(0, 0.5, n)
    predictions = {
        split: {"pIC50_BACE1": noisy[df["split"] == split]} for split in ("val", "test")
    }

    first, second, other = (
        pd.DataFrame(
            benchmark.evaluate_split_predictions(
                df, predictions, n_boot=200, boot_seed=seed
            )
        )
        for seed in (0, 0, 1)
    )

    pd.testing.assert_frame_equal(first, second)
    assert not first[["ci_low", "ci_high"]].equals(other[["ci_low", "ci_high"]])
    val, test = first[first["split"] == "val"], first[first["split"] == "test"]
    assert val["ci_low"].isna().all() and val["ci_high"].isna().all()
    assert (test["ci_low"] <= test["value"]).all()
    assert (test["value"] <= test["ci_high"]).all()
    rmse = test.set_index("metric").loc["rmse"]
    assert rmse["ci_low"] < rmse["ci_high"]


def test_run_benchmark_scores_val_only_and_promote_rebuilds_test(
    split_df, paths, tmp_path, monkeypatch
):
    path, data_path = paths
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "stub_default"
    promoted_path = tmp_path / "promoted.json"
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
        data_path=data_path,
        run_dir=run_dir,
    )

    assert seen_seeds == [0, 1, 2]
    # Training does not touch results.csv or score test_metrics.csv
    assert not path.exists()
    assert (run_dir / "val_metrics.csv").exists()
    assert not (run_dir / "test_metrics.csv").exists()
    assert set(summary["split"]) == {"val"}

    indexed = summary.set_index(["split", "metric"])
    assert (indexed["n_seeds"] == 3).all()
    assert indexed.loc[("val", "rmse"), "value"] == pytest.approx(0.2)
    assert indexed.loc[("val", "rmse"), "std"] == pytest.approx(0.2)

    saved = pd.read_csv(run_dir / "predictions" / "test_seed_2.csv")
    assert saved["smiles"].tolist() == ["CCCC", "CCCCC", "CCCCCC"]
    assert saved["pIC50_BACE1"].tolist()[:2] == pytest.approx([4.4, 8.4])

    # Promoting the run scores test once (val n_boot=0, test n_boot>0) and writes results.csv
    monkeypatch.setattr(
        benchmark, "load_benchmark_split", lambda *a, **kw: split_df.copy()
    )
    benchmark.promote_run(
        run_dir,
        tag="v1.1.0",
        promoted_path=promoted_path,
        runs_dir=runs_dir,
        results_path=path,
        data_path=data_path,
        n_boot=50,
    )
    assert json.loads(promoted_path.read_text())["tag"] == "v1.1.0"
    # Promoting without tag preserves existing tag
    benchmark.promote_run(
        run_dir,
        promoted_path=promoted_path,
        runs_dir=runs_dir,
        results_path=path,
        data_path=data_path,
        n_boot=50,
    )
    assert json.loads(promoted_path.read_text())["tag"] == "v1.1.0"
    # Failed rebuild rolls back promoted.json
    prev_promoted = promoted_path.read_text()
    with pytest.raises(FileNotFoundError):
        benchmark.promote_run(
            run_dir,
            tag="v9.9.9",
            promoted_path=promoted_path,
            runs_dir=tmp_path / "missing_runs",
            results_path=path,
            data_path=data_path,
            n_boot=50,
        )
    assert promoted_path.read_text() == prev_promoted

    assert (run_dir / "test_metrics.csv").exists()
    written = results.load_results(path)
    # 2 splits * 7 metrics * 3 seeds = 42 rows
    assert len(written) == 42
    assert written[written["split"] == "val"]["ci_low"].isna().all()
    assert written[written["split"] == "test"]["ci_low"].notna().any()
    assert (written["run_id"] == "stub_default").all()
    assert (written["params"] == '{"a": 1, "b": 2}').all()


def test_run_benchmark_rejects_empty_seeds(split_df, paths, tmp_path):
    _, data_path = paths
    with pytest.raises(ValueError, match="seeds must not be empty"):
        benchmark.run_benchmark(
            lambda df, seed: {},
            model="stub",
            encoder="enc",
            params={},
            seeds=(),
            df=split_df,
            data_path=data_path,
            run_dir=tmp_path / "run",
        )


def test_run_benchmark_loads_from_data_path_when_df_is_none(snapshot_csv, tmp_path):
    run_dir = tmp_path / "stub_default"
    seen_rows = []

    def fit_predict(df, seed):
        seen_rows.append(len(df))
        return {
            split: {
                "pIC50_BACE1": np.full((df["split"] == split).sum(), 6.0, dtype=float)
            }
            for split in ("val", "test")
        }

    benchmark.run_benchmark(
        fit_predict,
        model="stub",
        encoder="enc",
        params={},
        seeds=(0,),
        data_path=snapshot_csv,
        run_dir=run_dir,
    )

    assert seen_rows == [20]
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["data_sha256"] == results.file_sha256(snapshot_csv)


@pytest.mark.parametrize(
    "bad_predictions", [{}, {"val": {"pIC50_BACE1": [5.0, 7.0, np.nan]}}]
)
def test_run_benchmark_rejects_missing_split_predictions(
    split_df, paths, bad_predictions, tmp_path
):
    _, data_path = paths
    run_dir = tmp_path / "bad_run"
    with pytest.raises(ValueError, match="must return 'val' and 'test'"):
        benchmark.run_benchmark(
            lambda df, seed: bad_predictions,
            model="stub",
            encoder="enc",
            params={},
            seeds=(0,),
            df=split_df,
            data_path=data_path,
            run_dir=run_dir,
        )
    assert not run_dir.exists()


def test_crashed_run_stays_running_and_cannot_be_promoted(split_df, paths, tmp_path):
    path, data_path = paths
    run_dir = tmp_path / "runs" / "crashed"

    def fit_predict(df, seed):
        if seed == 1:
            raise RuntimeError("crash")
        return {
            "val": {"pIC50_BACE1": [5.0, 7.0, np.nan]},
            "test": {"pIC50_BACE1": [4.0, 8.0, np.nan]},
        }

    with pytest.raises(RuntimeError, match="crash"):
        benchmark.run_benchmark(
            fit_predict,
            model="stub",
            encoder="enc",
            params={},
            seeds=(0, 1),
            df=split_df,
            data_path=data_path,
            run_dir=run_dir,
        )

    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "running"
    assert meta["completed_seeds"] == [0]
    with pytest.raises(ValueError, match="incomplete run"):
        benchmark.promote_run(
            run_dir,
            promoted_path=tmp_path / "promoted.json",
            runs_dir=tmp_path / "runs",
            results_path=path,
            data_path=data_path,
        )


def test_load_benchmark_split_delegates_to_combined_and_scaffold(monkeypatch):
    sentinel = pd.DataFrame({"smiles": ["CCO"]})
    calls = []
    monkeypatch.setattr(benchmark, "load_combined", lambda **_: sentinel)

    def split(df, **kwargs):
        calls.append(kwargs)
        return df.assign(split=["train"])

    monkeypatch.setattr(benchmark, "scaffold_split", split)
    out = benchmark.load_benchmark_split()
    benchmark.load_benchmark_split(split_id="scaffold-3")
    assert out["split"].tolist() == ["train"]
    assert calls == [{}, {"seed": 3}]
    with pytest.raises(ValueError, match="split_id must be"):
        benchmark.load_benchmark_split(split_id="random-3")


def test_split_sha256_is_order_independent_and_changes_with_membership(split_df):
    reordered = split_df.iloc[::-1].set_axis(range(10, 16))
    assert benchmark.split_sha256(split_df) == benchmark.split_sha256(reordered)
    changed = split_df.copy()
    changed.loc[0, "split"] = "train"
    assert benchmark.split_sha256(split_df) != benchmark.split_sha256(changed)


def test_shared_split_is_reproducible_and_independent_of_training_seed(snapshot_csv):
    benchmark.seed_everything(0)
    first = benchmark.load_benchmark_split(snapshot_csv)
    benchmark.seed_everything(42)
    second = benchmark.load_benchmark_split(snapshot_csv)
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].tolist() == ["train"] * 16 + ["val"] * 2 + ["test"] * 2


def test_full_table_is_split_before_endpoint_filtering(snapshot_csv, monkeypatch):
    calls = []

    def split_full_table(df, **kwargs):
        calls.append((len(df), df["pIC50_BACE1"].isna().sum(), kwargs))
        return scaffold_split(df, **kwargs)

    monkeypatch.setattr(benchmark, "scaffold_split", split_full_table)
    full = benchmark.load_benchmark_split(snapshot_csv)
    endpoint = full.loc[full["pIC50_BACE1"].notna()]
    assert calls == [(20, 8, {})]
    assert endpoint["split"].value_counts().to_dict() == {
        "train": 8,
        "val": 2,
        "test": 2,
    }
    separate = scaffold_split(endpoint.drop(columns=["scaffold", "split"]))
    assert full.loc[16, "split"] == "val"
    assert separate.loc[16, "split"] == "train"


def test_load_is_read_only_and_missing_snapshot_raises(
    snapshot_csv, tmp_path, monkeypatch
):
    before = snapshot_csv.read_bytes()
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda *args, **kwargs: pytest.fail("benchmark loading must not write"),
    )
    benchmark.load_benchmark_split(snapshot_csv)
    assert snapshot_csv.read_bytes() == before

    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        benchmark.load_benchmark_split(missing)
    assert not missing.exists()


def test_frozen_benchmark_matches_recorded_rf_checksum_and_counts(frozen_benchmark):
    recorded = results.load_results()
    rf = recorded[recorded["model"].isin(["rf_bioactivity", "rf_toxicity"])]
    assert len(rf) == 124 * len(benchmark.SEEDS) * len(benchmark.SPLIT_IDS)
    assert set(rf["seed"]) == set(benchmark.SEEDS)
    assert set(rf["split_id"]) == set(benchmark.SPLIT_IDS)
    default = rf[rf["split_id"] == "default"]
    assert set(default["split_sha256"]) == {benchmark.split_sha256(frozen_benchmark)}
    original = "b340ddf73f38e10f8b6a2948156417661a269ffc9a6d9b097b538c4690e1b041"
    assert set(rf["data_sha256"]) == {original}
    assert results.file_sha256(COMBINED_CSV_PATH) == original

    for row in default[default["n_labeled"].notna()].itertuples(index=False):
        labels = frozen_benchmark.loc[
            frozen_benchmark["split"] == row.split, row.endpoint
        ].dropna()
        assert len(labels) == row.n_labeled
        positive = (
            labels >= PIC50_ACTIVE_THRESHOLD
            if row.model == "rf_bioactivity"
            else labels == 1
        )
        assert positive.sum() == row.n_positive
