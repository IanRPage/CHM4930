import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import benchmark as bm
import encoder_benchmark as eb
from baseline import results

AI = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch):
    module = import_path(
        "smiles_sweep_test", AI / "scripts" / "SMILES_encoder_sweep.py"
    )
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    return module


@pytest.fixture(autouse=True)
def protect_results_and_random_state(monkeypatch):
    before = results.RESULTS_PATH.read_bytes()
    record = bm.record_results
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def record_copy_only(*args, **kwargs):
        assert (
            kwargs.get("path", results.RESULTS_PATH).resolve()
            != results.RESULTS_PATH.resolve()
        )
        return record(*args, **kwargs)

    monkeypatch.setattr(bm, "record_results", record_copy_only)
    yield
    assert results.RESULTS_PATH.read_bytes() == before
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.set_num_threads(threads)


@pytest.fixture
def data(tmp_path):
    frame = pd.DataFrame(
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
    frame.to_csv(path, index=False)
    return eb.BACE1Benchmark.load(path)


@pytest.fixture(scope="module")
def frozen_data():
    return eb.BACE1Benchmark.load()


@pytest.fixture
def mocked_pretrained(monkeypatch):
    """Load the real wrapper with a scoped Transformers stand-in, without downloads."""
    calls, encoded, models, initial = [], [], [], []

    class Tokenizer:
        def __call__(self, smiles, **kwargs):
            encoded.extend(smiles)
            input_ids = torch.tensor([[len(s), sum(map(ord, s)) % 17] for s in smiles])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    class Model(nn.Module):
        def __init__(self, revision):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, _commit_hash=revision)
            self.projection = nn.Linear(2, 4)

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(pooler_output=self.projection(input_ids.float()))

    def tokenizer_from_pretrained(model_name, **kwargs):
        calls.append(("tokenizer", model_name, kwargs))
        return Tokenizer()

    def model_from_pretrained(model_name, **kwargs):
        calls.append(("model", model_name, kwargs))
        model = Model(kwargs.get("revision"))
        models.append(model)
        initial.append(copy.deepcopy(model.state_dict()))
        return model

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=tokenizer_from_pretrained
    )
    transformers.AutoModel = SimpleNamespace(from_pretrained=model_from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    module = import_path(
        "mocked_smiles_encoder", AI / "src" / "encoders" / "smiles_encoder.py"
    )
    monkeypatch.setitem(sys.modules, "encoders.smiles_encoder", module)
    return SimpleNamespace(
        module=module, calls=calls, encoded=encoded, models=models, initial=initial
    )


def config(script, **overrides):
    return script.normalize_config(
        {
            "PRETRAINED_REVISION": REVISION,
            "EPOCHS": 2,
            "BATCH_SIZE": 4,
            "ENCODER_BATCH_SIZE": 3,
            **overrides,
        }
    )


def fake_measurements(cfg):
    return {
        "best_epoch": 1,
        "device": "cpu",
        "pretrained": {
            "model_name": cfg["MODEL_NAME"],
            "revision": cfg["PRETRAINED_REVISION"],
            "resolved_revision": cfg["PRETRAINED_REVISION"],
            "freeze": True,
            "head": "linear",
        },
        **{
            f"{split}_{metric}": 0.5
            for split in ("train", "val", "test")
            for metric in eb.METRICS
        },
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"PRETRAINED_REVISION": "main"},
        {"PRETRAINED_REVISION": None},
        {"FREEZE": False},
        {"BATCH_SIZE": 0},
        {"LR": float("nan")},
    ],
)
def test_requires_pinned_frozen_valid_configuration(script, bad):
    with pytest.raises(ValueError):
        config(script, **bad)


def test_wrapper_forwards_revision_and_defaults_to_pinned_revision(
    mocked_pretrained,
):
    mocked_pretrained.module.SmilesEncoder(revision=REVISION)
    assert all(call[2]["revision"] == REVISION for call in mocked_pretrained.calls)
    assert mocked_pretrained.calls[0][2]["trust_remote_code"]
    assert mocked_pretrained.calls[1][2]["deterministic_eval"]
    mocked_pretrained.calls.clear()
    mocked_pretrained.module.SmilesEncoder()
    assert all(
        call[2]["revision"] == mocked_pretrained.module.DEFAULT_REVISION
        for call in mocked_pretrained.calls
    )


@pytest.mark.parametrize("evaluate_test", [False, True])
def test_real_wrapper_and_head_preserve_order_and_frozen_weights(
    script, data, mocked_pretrained, monkeypatch, evaluate_test
):
    def unexpected_features(*args, **kwargs):
        raise AssertionError("SMILES runs must not build fingerprints or graphs")

    monkeypatch.setattr(data, "features", unexpected_features)
    observed, original_score = [], data.score
    training_labels = []
    original_loss = nn.functional.mse_loss

    def training_loss(predictions, targets, **kwargs):
        training_labels.extend(targets.detach().cpu().tolist())
        return original_loss(predictions, targets, **kwargs)

    monkeypatch.setattr(nn.functional, "mse_loss", training_loss)

    def score(predict, *, evaluate_test=False):
        val = data.indices["val"]
        assert torch.equal(predict(val).flip(0), predict(val[::-1].copy()))

        def aligned(indices):
            observed.append(indices.tolist())
            return predict(indices)

        return original_score(aligned, evaluate_test=evaluate_test)

    monkeypatch.setattr(data, "score", score)
    result = script.run(config(script), 42, data, evaluate_test=evaluate_test)
    names = ("train", "val", "test") if evaluate_test else ("train", "val")
    assert observed == [data.indices[name].tolist() for name in names]
    assert sorted(training_labels) == sorted(data.y[data.indices["train"]].tolist() * 2)
    expected_smiles = [
        smiles
        for name in names
        for smiles in data.frame.iloc[data.indices[name]]["smiles"]
    ]
    assert mocked_pretrained.encoded == expected_smiles
    assert result["pretrained"]["revision"] == REVISION
    assert result["pretrained"]["freeze"] is True
    model = mocked_pretrained.models[0]
    assert not model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, mocked_pretrained.initial[0][name])
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert (
        np.isfinite(result["test_RMSE"])
        if evaluate_test
        else np.isnan(result["test_RMSE"])
    )


def test_checkpoint_selection_restores_validation_best_before_encoding_test(
    script, data, mocked_pretrained, monkeypatch
):
    val_predictions = []

    def validation_metric(labels, predictions, task_type):
        pd.testing.assert_series_equal(
            labels, data.frame.iloc[data.indices["val"]]["pIC50_BACE1"]
        )
        val_predictions.append(predictions.clone())
        return {"rmse": 0.1 if len(val_predictions) == 1 else 100.0}

    monkeypatch.setattr(script, "evaluate_predictions", validation_metric)
    original_score = data.score

    def selected_score(predict, *, evaluate_test=False):
        assert len(val_predictions) == 2
        test_smiles = set(data.frame.iloc[data.indices["test"]]["smiles"])
        assert not test_smiles.intersection(mocked_pretrained.encoded)
        assert torch.equal(predict(data.indices["val"]), val_predictions[0])
        return original_score(predict, evaluate_test=evaluate_test)

    monkeypatch.setattr(data, "score", selected_score)
    result = script.run(config(script), 42, data, evaluate_test=True)
    assert result["best_epoch"] == 1


def test_test_labels_do_not_affect_training_or_selection(
    script, data, mocked_pretrained
):
    first = script.run(config(script), 42, data, evaluate_test=True)
    changed = copy.deepcopy(data)
    changed.frame.loc[changed.indices["test"], "pIC50_BACE1"] += 1000
    second = script.run(config(script), 42, changed, evaluate_test=True)
    assert first["best_epoch"] == second["best_epoch"]
    for name in ("train_RMSE", "val_RMSE", "val_R2", "val_ROC-AUC"):
        assert first[name] == second[name]
    assert first["test_RMSE"] != second["test_RMSE"]


def test_embeddings_follow_requested_smiles_order(script, data, mocked_pretrained):
    cfg = config(script)
    encoder = script.make_encoder(cfg).eval()
    smiles = data.frame["smiles"].tolist()
    first = encoder(smiles)
    reversed_scores = encoder(smiles[::-1])
    assert torch.equal(first.flip(0), reversed_scores)
    assert first.shape == (len(data.frame), encoder.out_dim)


def test_mismatched_loaded_revision_is_rejected(
    script, data, mocked_pretrained, monkeypatch
):
    original = script.make_encoder

    def wrong_revision(cfg):
        encoder = original(cfg)
        encoder.model.config._commit_hash = "b" * 40
        return encoder

    monkeypatch.setattr(script, "make_encoder", wrong_revision)
    with pytest.raises(ValueError, match="revision does not match"):
        script.run(config(script), 42, data)


def test_real_canonical_cohort_and_result_format_preserve_all_rf_rows_on_copy(
    script, frozen_data, tmp_path, monkeypatch
):
    assert {split: len(idx) for split, idx in frozen_data.indices.items()} == {
        "train": 6469,
        "val": 808,
        "test": 810,
    }
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps(config(script)))
    copy_path = tmp_path / "results.csv"
    copy_path.write_bytes(results.RESULTS_PATH.read_bytes())
    before = pd.read_csv(copy_path)
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: frozen_data)
    calls = []

    def run(cfg, seed, cohort, *, evaluate_test=False):
        calls.append((seed, cohort, evaluate_test))
        return fake_measurements(cfg)

    monkeypatch.setattr(script, "run", run)
    record = bm.record_results
    monkeypatch.setattr(
        bm,
        "record_results",
        lambda rows, **kwargs: record(rows, path=copy_path, **kwargs),
    )
    rows = script.main(["--benchmark", "--config", str(chosen), "--write-results"])
    assert calls == [(42, frozen_data, True)]
    assert len(rows) == 6
    for row in rows:
        assert row["dataset"] == "bace1" and row["endpoint"] == "pIC50_BACE1"
        assert row["n_labeled"] == (808 if row["split"] == "val" else 810)
        assert row["n_positive"] == (484 if row["split"] == "val" else 517)
        params = json.loads(row["params"])
        assert params["benchmark"] == frozen_data.metadata
        assert params["seed"] == 42
        assert params["score_source"] == "predicted_pic50"
        assert params["pretrained"]["revision"] == REVISION
        assert params["pretrained"]["freeze"] is True
    assert {row["metric"] for row in rows} == {"rmse", "r2", "roc_auc"}
    after = pd.read_csv(copy_path)
    pd.testing.assert_frame_equal(
        before,
        after.loc[after["model"] != "smiles_encoder"].reset_index(drop=True),
        rtol=1e-14,
        atol=1e-14,
    )
    assert after.columns.tolist() == results.COLUMNS
    assert len(after) == 52


def test_dry_run_needs_no_transformers_weights_or_writes(
    script, frozen_data, tmp_path, monkeypatch
):
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps(config(script)))
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: frozen_data)
    monkeypatch.setattr(
        script,
        "make_encoder",
        lambda *args: pytest.fail("dry run must not load weights"),
    )
    assert (
        script.main(
            ["--benchmark", "--config", str(chosen), "--dry-run", "--write-results"]
        )
        == []
    )


def test_hyperparameter_selection_uses_only_validation(
    script, frozen_data, tmp_path, monkeypatch
):
    records = []
    for lr, val_rmse, test_rmse in ((0.001, 0.5, 100.0), (0.01, 1.0, 0.01)):
        for seed in (0, 1, 2):
            records.append(
                {
                    "cfg": eb.config_json(config(script, LR=lr)),
                    "seed": seed,
                    "val_RMSE": val_rmse,
                    "test_RMSE": test_rmse,
                    "benchmark": eb.config_json(frozen_data.metadata),
                    "sweep_protocol": eb.SWEEP_PROTOCOL,
                    "experiment": "smiles_encoder",
                }
            )
    path = tmp_path / "stage.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: frozen_data)
    calls = []

    def run(cfg, seed, cohort, *, evaluate_test=False):
        calls.append((cfg["LR"], seed, evaluate_test))
        return fake_measurements(cfg)

    monkeypatch.setattr(script, "run", run)
    script.main(["--benchmark", "--select-from", str(path)])
    assert calls == [(0.001, 42, True)]


def test_exploratory_stage_config_and_resume_are_shared(
    script, data, tmp_path, monkeypatch
):
    path = tmp_path / "pretrained.json"
    path.write_text(json.dumps({"PRETRAINED_REVISION": REVISION, "EPOCHS": 2}))
    output = tmp_path / "sweep"
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: data)
    calls = []

    def run(cfg, seed, cohort, *, evaluate_test=False):
        assert not evaluate_test
        calls.append((cfg["LR"], seed))
        return {
            **fake_measurements(cfg),
            "cfg": eb.config_json(cfg),
            "seed": seed,
            "secs": 0.1,
            **{f"test_{metric}": np.nan for metric in eb.METRICS},
        }

    monkeypatch.setattr(script, "run", run)
    argv = ["--stage-config", str(path), "--output-dir", str(output)]
    script.main(argv)
    assert len(calls) == 9
    script.main(argv)
    assert len(calls) == 9
    loaded = eb.load_sweep_records(output / "1-head.csv", data, "smiles_encoder")
    assert loaded["test_RMSE"].isna().all()
    assert len(loaded) == 9
