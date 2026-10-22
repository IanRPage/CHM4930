import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import benchmark as bm
import encoder_benchmark as eb
from baseline import results
from pipeline.combined_loader import COMBINED_CSV_PATH, load_combined
from pipeline.preprocess import featurize
from pipeline.split import scaffold_split

AI = Path(__file__).resolve().parents[1]
SCRIPTS = ["ECFP4_encoder_sweep", "gnn_encoder_sweep"]


def import_script(name):
    spec = importlib.util.spec_from_file_location(name, AI / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def forbid_tracked_results_writes(monkeypatch):
    record = bm.record_results

    def guarded_record(*args, **kwargs):
        assert (
            kwargs.get("path", results.RESULTS_PATH).resolve()
            != results.RESULTS_PATH.resolve()
        )
        return record(*args, **kwargs)

    monkeypatch.setattr(bm, "record_results", guarded_record)


@pytest.fixture
def data(tmp_path):
    table = pd.DataFrame(
        {
            "smiles": ["C1" + "C" * (n - 1) + "1" for n in range(3, 23)],
            "pIC50_BACE1": [
                5.123456789 + i % 4 if i < 8 or i >= 16 else np.nan for i in range(20)
            ],
            "pIC50_EGFR": 6.0,
            "in_bace1": True,
            "in_egfr": True,
        }
    )
    path = tmp_path / "combined.csv"
    table.to_csv(path, index=False)
    return eb.BACE1Benchmark.load(path)


@pytest.fixture(scope="module")
def frozen_data():
    before = {
        path: path.read_bytes() for path in (COMBINED_CSV_PATH, results.RESULTS_PATH)
    }
    yield eb.BACE1Benchmark.load()
    for path, contents in before.items():
        assert path.read_bytes() == contents


@pytest.fixture
def cpu_training(monkeypatch):
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.set_num_threads(threads)


def measurements(config=None, seed=42):
    return {
        "cfg": eb.config_json(config or {}),
        "seed": seed,
        "best_epoch": 1,
        "secs": 0.1,
        "device": "cpu",
        "deterministic_algorithms": False,
        **{
            f"{split}_{metric}": 0.5
            for split in ("train", "val", "test")
            for metric in eb.METRICS
        },
    }


def records(data, experiment="ecfp4_encoder", configs=None, seeds=(0, 1, 2)):
    configs = configs or [{"LR": 0.001}, {"LR": 0.01}]
    return pd.DataFrame(
        [
            {
                **measurements(config, seed),
                "val_RMSE": 1.0 + position,
                "test_RMSE": 100.0 - position,
                "benchmark": eb.config_json(data.metadata),
                "sweep_protocol": eb.SWEEP_PROTOCOL,
                "experiment": experiment,
            }
            for position, config in enumerate(configs)
            for seed in seeds
        ]
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_import_has_no_dataset_featurization_or_write_side_effects(script, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("imports must not load data, featurize, or write")

    monkeypatch.setattr(eb.BACE1Benchmark, "load", unexpected)
    monkeypatch.setattr(eb, "featurize_many", unexpected)
    monkeypatch.setattr(bm, "record_results", unexpected)
    module = import_script(script)
    assert module.STAGES


def test_encoder_splits_full_table_before_filtering(data):
    assert len(data.full) == 20
    assert len(data.frame) == 12
    assert {name: len(idx) for name, idx in data.indices.items()} == {
        "train": 8,
        "val": 2,
        "test": 2,
    }
    repeated = eb.BACE1Benchmark.load(data.data_path)
    pd.testing.assert_frame_equal(repeated.frame, data.frame)
    separately_split = scaffold_split(
        load_combined(path=data.data_path).dropna(subset=["pIC50_BACE1"])
    )
    assert set(data.frame.loc[data.indices["val"], "smiles"]) != set(
        separately_split.loc[separately_split["split"] == "val", "smiles"]
    )
    assert data.metadata["n_molecules"] == 20
    assert data.metadata == bm.benchmark_metadata(data.full, data.data_path)


def test_real_encoder_cohorts_match_bace1_rf_exactly(frozen_data):
    rf_full = scaffold_split(load_combined())
    for split, count in (("train", 6469), ("val", 808), ("test", 810)):
        rf = rf_full.loc[(rf_full["split"] == split) & rf_full["pIC50_BACE1"].notna()]
        encoder = frozen_data.frame.iloc[frozen_data.indices[split]]
        assert encoder["smiles"].tolist() == rf["smiles"].tolist()
        np.testing.assert_array_equal(encoder["pIC50_BACE1"], rf["pIC50_BACE1"])
        assert len(encoder) == count
    recorded = pd.read_csv(results.RESULTS_PATH)
    for row in recorded.loc[
        (recorded["model"] == "rf_bioactivity") & (recorded["dataset"] == "bace1")
    ].itertuples():
        cohort = frozen_data.frame.iloc[frozen_data.indices[row.split]]
        assert len(cohort) == row.n_labeled
        assert cohort["active_BACE1"].eq(1).sum() == row.n_positive


def test_fingerprints_graphs_and_targets_remain_aligned(data):
    for n_bits in (1024, 2048):
        fingerprints = data.fingerprints(n_bits)
        assert fingerprints.shape == (len(data.frame), n_bits)
        for position, smiles in enumerate(data.frame["smiles"]):
            assert torch.equal(fingerprints[position], featurize(smiles, n_bits).fp[0])
    graphs = data.graphs()
    assert graphs is data.graphs()
    assert [graph.smiles for graph in graphs] == data.frame["smiles"].tolist()
    assert torch.equal(torch.cat([graph.y for graph in graphs]), data.y)


@pytest.mark.parametrize("failure", ["drop", "reorder"])
def test_featurization_cannot_silently_change_the_cohort(data, monkeypatch, failure):
    graphs = [featurize(smiles) for smiles in data.frame["smiles"]]
    bad = (graphs[:-1], [len(graphs) - 1]) if failure == "drop" else (graphs[::-1], [])
    monkeypatch.setattr(eb, "featurize_many", lambda *args: bad)
    with pytest.raises(ValueError, match="cohort|identity or row order"):
        data.graphs()


@pytest.mark.parametrize("evaluate_test", [False, True])
def test_scoring_uses_aligned_original_labels_and_test_only_when_requested(
    data, evaluate_test
):
    calls = []

    def predict(indices):
        calls.append(indices.tolist())
        return data.frame.iloc[indices]["pIC50_BACE1"].to_numpy()

    scored = data.score(predict, evaluate_test=evaluate_test)
    assert calls == [
        data.indices[name].tolist()
        for name in ("train", "val", "test")[: 3 if evaluate_test else 2]
    ]
    assert scored["val_RMSE"] == 0
    assert scored["val_R2"] == 1
    assert scored["val_ROC-AUC"] == 1
    assert scored["test_RMSE"] == 0 if evaluate_test else np.isnan(scored["test_RMSE"])


def test_scoring_rejects_misordered_pandas_predictions(data):
    with pytest.raises(ValueError, match="prediction row index"):
        data.score(lambda idx: data.frame.iloc[idx]["pIC50_BACE1"].iloc[::-1])


def test_configuration_selection_ignores_test_metrics(data):
    table = records(data)
    config, selection = eb.select_configuration(table, [0, 1, 2])
    assert config == {"LR": 0.001}
    assert selection == {
        "method": "mean_validation_rmse",
        "seeds": [0, 1, 2],
        "mean_val_rmse": 1.0,
    }
    table["test_RMSE"] = [0, -1000, np.nan, 200, 300, 400]
    assert eb.select_configuration(table, [0, 1, 2]) == (config, selection)


def test_selection_requires_complete_matching_seed_sets_and_rejects_conflicts(data):
    table = records(data)
    incomplete = table.drop(index=2)
    assert eb.select_configuration(incomplete, [0, 1, 2])[0] == {"LR": 0.01}
    with pytest.raises(ValueError, match="complete selection seed set"):
        eb.select_configuration(table.iloc[:2], [0, 1, 2])
    duplicate = table.iloc[:1].assign(val_RMSE=0.1)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        eb.select_configuration(pd.concat([table, duplicate]), [0, 1, 2])


def test_compatible_resume_normalizes_config_json(data, tmp_path):
    table = records(data, configs=[{"LR": 0.01, "EPOCHS": 10}])
    table["cfg"] = '{"LR": 0.01, "EPOCHS": 10}'
    path = tmp_path / "stage.csv"
    table.to_csv(path, index=False)
    loaded = eb.load_sweep_records(path, data, "ecfp4_encoder")
    assert loaded["cfg"].eq(eb.config_json({"EPOCHS": 10, "LR": 0.01})).all()
    assert len(loaded) == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_sha256", "old"),
        ("split_sha256", "old"),
        ("split_fractions", [0.6, 0.2, 0.2]),
        ("split_seed", 42),
        ("benchmark_version", 0),
        ("training_seed", 0),
    ],
)
def test_resume_rejects_incompatible_benchmark_without_writing(
    data, tmp_path, field, value
):
    table = records(data)
    metadata = data.metadata | {field: value}
    table.loc[0, "benchmark"] = eb.config_json(metadata)
    path = tmp_path / "stage.csv"
    table.to_csv(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match=field):
        eb.load_sweep_records(path, data, "ecfp4_encoder")
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sweep_protocol", "old", "protocol"),
        ("experiment", "gnn_encoder", "experiment"),
        ("seed", -1, "training seeds"),
        ("val_RMSE", np.inf, "finite"),
        ("cfg", "[]", "JSON object"),
    ],
)
def test_resume_rejects_other_incompatible_records(
    data, tmp_path, column, value, message
):
    table = records(data)
    table.loc[0, column] = value
    path = tmp_path / "stage.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        eb.load_sweep_records(path, data, "ecfp4_encoder")


@pytest.mark.parametrize("script", SCRIPTS)
def test_legacy_csv_and_logs_are_preserved_and_not_resumed(
    script, data, tmp_path, monkeypatch
):
    module = import_script(script)
    name = next(iter(module.STAGES))
    path = tmp_path / f"{name}.csv"
    log = path.with_suffix(".log")
    pd.DataFrame([measurements()]).to_csv(path, index=False)
    log.write_text("historical run\n")
    before = (path.read_bytes(), log.read_bytes())
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: data)
    monkeypatch.setattr(
        module,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "legacy records must be rejected before training"
        ),
    )
    with pytest.raises(ValueError, match="legacy/incompatible.*preserve this file"):
        module.main(["1", "--output-dir", str(tmp_path)])
    assert (path.read_bytes(), log.read_bytes()) == before


@pytest.mark.parametrize("script", SCRIPTS)
def test_exploratory_stage_csv_and_log_resume_without_test_evaluation(
    script, data, tmp_path, monkeypatch
):
    module = import_script(script)
    cfg = {"LR": 0.001}
    monkeypatch.setattr(module, "STAGES", {"1-tiny": ([cfg], [0])})
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: data)
    calls = []

    def run(config, seed, cohort, *, evaluate_test=False):
        assert not evaluate_test
        calls.append((config, seed))
        result = measurements(config, seed)
        result.update({f"test_{metric}": np.nan for metric in eb.METRICS})
        return result

    monkeypatch.setattr(module, "run", run)
    module.main(["1", "--output-dir", str(tmp_path)])
    path, log = tmp_path / "1-tiny.csv", tmp_path / "1-tiny.log"
    before = (path.read_bytes(), log.read_bytes())
    module.main(["1", "--output-dir", str(tmp_path)])
    assert calls == [(cfg, 0)]
    assert (path.read_bytes(), log.read_bytes()) == before
    table = pd.read_csv(path)
    assert set(measurements()) <= set(table.columns)
    assert table["test_RMSE"].isna().all()
    assert (
        len(
            eb.load_sweep_records(
                path,
                data,
                "ecfp4_encoder" if script.startswith("ECFP4") else "gnn_encoder",
            )
        )
        == 1
    )


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("dry_run", [False, True])
def test_canonical_seed42_and_explicit_export_opt_in(
    script, dry_run, frozen_data, tmp_path, monkeypatch
):
    module = import_script(script)
    path = tmp_path / "chosen.json"
    path.write_text('{"EPOCHS": 2}')
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: frozen_data)
    calls = []

    def run(config, seed, cohort, *, evaluate_test=False):
        assert seed == 42
        assert cohort is frozen_data
        assert evaluate_test
        calls.append(config)
        return measurements(config, seed)

    monkeypatch.setattr(module, "run", run)
    argv = ["--benchmark", "--config", str(path)]
    if dry_run:
        argv += ["--dry-run", "--write-results"]
    rows = module.main(argv)
    assert len(calls) == (0 if dry_run else 1)
    assert len(rows) == (0 if dry_run else 6)
    if rows:
        assert {row["metric"] for row in rows} == {"rmse", "r2", "roc_auc"}
        for row in rows:
            params = json.loads(row["params"])
            assert params["seed"] == 42
            assert params["benchmark"] == frozen_data.metadata
            assert params["best_epoch"] == 1
            assert row["n_labeled"] == (808 if row["split"] == "val" else 810)
            assert row["n_positive"] == (484 if row["split"] == "val" else 517)


@pytest.mark.parametrize("script", SCRIPTS)
def test_explicit_canonical_export_preserves_rf_schema_and_rows_on_copy(
    script, frozen_data, tmp_path, monkeypatch
):
    module = import_script(script)
    chosen = tmp_path / "chosen.json"
    chosen.write_text("{}")
    path = tmp_path / "results.csv"
    path.write_bytes(results.RESULTS_PATH.read_bytes())
    before = pd.read_csv(path)
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: frozen_data)
    monkeypatch.setattr(
        module, "run", lambda config, seed, cohort, **kwargs: measurements(config, seed)
    )
    record = bm.record_results

    def record_copy(rows, **kwargs):
        return record(rows, path=path, **kwargs)

    monkeypatch.setattr(bm, "record_results", record_copy)
    module.main(["--benchmark", "--config", str(chosen), "--write-results"])
    after = pd.read_csv(path)
    rf = after[after["model"].isin(["rf_bioactivity", "rf_toxicity"])].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(before, rf, rtol=1e-14, atol=1e-14)
    assert len(rf) == 46
    assert len(after) == 52
    assert after.columns.tolist() == results.COLUMNS
    assert set(after["model"]) == {
        "rf_bioactivity",
        "rf_toxicity",
        "ecfp4_encoder" if script.startswith("ECFP4") else "gnn_encoder",
    }


def test_export_refuses_to_discard_rf_from_another_snapshot(data, tmp_path):
    path = tmp_path / "results.csv"
    path.write_bytes(results.RESULTS_PATH.read_bytes())
    original = path.read_bytes()
    with pytest.raises(ValueError, match="discard RF results"):
        eb.verify_export(data, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("script", SCRIPTS)
def test_canonical_selection_finishes_before_seed42_test_run(
    script, frozen_data, tmp_path, monkeypatch
):
    module = import_script(script)
    experiment = "ecfp4_encoder" if script.startswith("ECFP4") else "gnn_encoder"
    path = tmp_path / "stage.csv"
    records(frozen_data, experiment).to_csv(path, index=False)
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: frozen_data)
    calls = []

    def run(config, seed, cohort, *, evaluate_test=False):
        calls.append((config["LR"], seed, evaluate_test))
        return measurements(config, seed)

    monkeypatch.setattr(module, "run", run)
    module.main(["--benchmark", "--select-from", str(path)])
    assert calls == [(0.001, 42, True)]


@pytest.mark.parametrize("script", SCRIPTS)
def test_best_epoch_restored_before_test_scoring(
    script, data, monkeypatch, cpu_training
):
    module = import_script(script)
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    cfg = module.normalize_config(
        {"HIDDEN_DIMS": [8], "EPOCHS": 2, "BATCH_SIZE": 4, "DROPOUT": 0.0}
    )
    if script.startswith("ECFP4"):
        cfg["N_BITS"] = 1024
    validation_predictions = []
    events = []

    def validation_metric(labels, predictions, task_type):
        pd.testing.assert_series_equal(
            labels, data.frame.iloc[data.indices["val"]]["pIC50_BACE1"]
        )
        assert task_type == "regression"
        validation_predictions.append(predictions.clone())
        events.append("validation")
        return {"rmse": 0.1 if len(validation_predictions) == 1 else 100.0}

    monkeypatch.setattr(module, "evaluate_predictions", validation_metric)
    original_score = data.score

    def score_selected(predict, *, evaluate_test=False):
        assert events == ["validation", "validation"]
        assert evaluate_test
        assert torch.equal(predict(data.indices["val"]), validation_predictions[0])
        events.append("selected")
        return original_score(predict, evaluate_test=evaluate_test)

    monkeypatch.setattr(data, "score", score_selected)
    trained = module.run(cfg, 42, data, evaluate_test=True)
    assert trained["best_epoch"] == 1
    assert np.isfinite(trained["test_RMSE"])
    assert events == ["validation", "validation", "selected"]


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_gnn_normal_batching_keeps_existing_order_across_epochs(batch_size):
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler

    module = import_script("gnn_encoder_sweep")

    def batches(safe):
        generator = torch.Generator().manual_seed(42)
        sampler = (module.TrainingBatchSampler if safe else BatchSampler)(
            RandomSampler(range(260), generator=generator), batch_size, drop_last=False
        )
        loader = DataLoader(range(260), batch_sampler=sampler, generator=generator)
        return [[batch.tolist() for batch in loader] for _ in range(2)]

    assert batches(True) == batches(False)


def test_gnn_singleton_training_tail_keeps_all_rows_and_evaluation_samples(
    data, monkeypatch, cpu_training
):
    module = import_script("gnn_encoder_sweep")
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    sizes = []
    original = module.GNNEncoder

    class ObservedEncoder(original):
        def forward(self, graph):
            sizes.append((self.training, graph.num_graphs))
            return super().forward(graph)

    monkeypatch.setattr(module, "GNNEncoder", ObservedEncoder)
    cfg = module.normalize_config(
        {"HIDDEN_DIMS": [8], "BATCH_SIZE": 7, "EPOCHS": 2, "DROPOUT": 0.0}
    )
    sampler = module.TrainingBatchSampler(range(8), 7, drop_last=False)
    assert list(sampler) == [list(range(8))]
    assert len(sampler) == 1
    result = module.run(cfg, 42, data, evaluate_test=True)
    assert [size for training, size in sizes if training] == [8, 8]
    assert (False, 1) in sizes  # The final evaluation batch must not be dropped.
    assert np.isfinite(result["test_RMSE"])


@pytest.mark.parametrize(
    "enabled,warn_only", [(False, False), (True, False), (True, True), (False, True)]
)
@pytest.mark.parametrize("fail", [False, True])
def test_gnn_restores_deterministic_settings_on_success_and_failure(
    data, monkeypatch, cpu_training, enabled, warn_only, fail
):
    module = import_script("gnn_encoder_sweep")
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    observed = []
    original_predict = module.predict

    def predict(*args):
        observed.append(
            (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
        )
        if fail:
            raise RuntimeError("synthetic prediction failure")
        return original_predict(*args)

    monkeypatch.setattr(module, "predict", predict)
    cfg = module.normalize_config(
        {"HIDDEN_DIMS": [8], "EPOCHS": 1, "BATCH_SIZE": 4, "DROPOUT": 0.0}
    )
    if fail:
        with pytest.raises(RuntimeError, match="synthetic prediction failure"):
            module.run(cfg, 42, data)
    else:
        result = module.run(cfg, 42, data)
        assert result["deterministic_algorithms"] is True
    assert observed and all(state == (True, False) for state in observed)
    assert torch.are_deterministic_algorithms_enabled() == enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
