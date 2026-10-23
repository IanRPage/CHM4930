import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import encoder_benchmark as eb
import encoders.smiles_encoder as smiles_module
from baseline import benchmark as bm
from pipeline.preprocess import featurize

AI_DIR = Path(__file__).resolve().parents[2]
SCRIPTS = ["ECFP4_encoder_sweep", "gnn_encoder_sweep", "SMILES_encoder_sweep"]


def import_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, AI_DIR / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measurements(config=None, seed=42):
    return {
        "cfg": bm.params_json(config or {}),
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
                "data_sha256": data.data_sha256,
                "split_sha256": data.split_sha256,
                "sweep_protocol": eb.SWEEP_PROTOCOL,
                "experiment": experiment,
            }
            for position, config in enumerate(configs)
            for seed in seeds
        ]
    )


@pytest.fixture
def mocked_smiles_encoder(monkeypatch):
    encoded, models, initial = [], [], []

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

    monkeypatch.setattr(
        smiles_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )

    def load_model(*args, **kwargs):
        model = Model(kwargs.get("revision"))
        models.append(model)
        initial.append(copy.deepcopy(model.state_dict()))
        return model

    monkeypatch.setattr(smiles_module.AutoModel, "from_pretrained", load_model)
    return SimpleNamespace(encoded=encoded, models=models, initial=initial)


@pytest.mark.parametrize("script", SCRIPTS)
def test_import_has_no_dataset_featurization_or_write_side_effects(script, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("imports must not load data, featurize, or write")

    monkeypatch.setattr(eb.BACE1Benchmark, "load", unexpected)
    monkeypatch.setattr(eb, "featurize_many", unexpected)
    monkeypatch.setattr(bm, "record_results", unexpected)
    module = import_script(script)
    assert module.STAGES


def test_bace1_cohorts_and_featurization_alignment(
    bace1_data, frozen_bace1, monkeypatch
):
    assert len(bace1_data.full) == 20
    assert len(bace1_data.frame) == 12
    assert {name: len(idx) for name, idx in bace1_data.indices.items()} == {
        "train": 8,
        "val": 2,
        "test": 2,
    }
    assert {name: len(idx) for name, idx in frozen_bace1.indices.items()} == {
        "train": 6469,
        "val": 808,
        "test": 810,
    }

    for n_bits in (1024, 2048):
        fps = bace1_data.fingerprints(n_bits)
        assert fps.shape == (len(bace1_data.frame), n_bits)
        assert torch.equal(
            fps[0], featurize(bace1_data.frame["smiles"].iloc[0], n_bits).fp[0]
        )
    graphs = bace1_data.graphs()
    assert graphs is bace1_data.graphs()
    assert [g.smiles for g in graphs] == bace1_data.frame["smiles"].tolist()
    assert torch.equal(torch.cat([g.y for g in graphs]), bace1_data.y)

    for bad in ((graphs[:-1], [len(graphs) - 1]), (graphs[::-1], [])):
        bace1_data._features.clear()
        monkeypatch.setattr(eb, "featurize_many", lambda *args, b=bad: b)
        with pytest.raises(ValueError, match="cohort|identity or row order"):
            bace1_data.graphs()


@pytest.mark.parametrize("evaluate_test", [False, True])
def test_scoring_alignment_and_optional_test_evaluation(bace1_data, evaluate_test):
    calls = []

    def predict(indices):
        calls.append(indices.tolist())
        return bace1_data.frame.iloc[indices]["pIC50_BACE1"].to_numpy()

    scored = bace1_data.score(predict, evaluate_test=evaluate_test)
    assert calls == [
        bace1_data.indices[name].tolist()
        for name in ("train", "val", "test")[: 3 if evaluate_test else 2]
    ]
    assert (
        scored["val_RMSE"] == 0 and scored["val_R2"] == 1 and scored["val_ROC-AUC"] == 1
    )
    assert scored["test_RMSE"] == 0 if evaluate_test else np.isnan(scored["test_RMSE"])

    with pytest.raises(ValueError, match="expected 1D array of length"):
        bace1_data.score(
            lambda idx: bace1_data.frame.iloc[idx[:-1]]["pIC50_BACE1"].to_numpy()
        )


def test_configuration_selection_requires_complete_seeds_and_ignores_test(bace1_data):
    table = records(bace1_data)
    config, selection = eb.select_configuration(table, [0, 1, 2])
    assert config == {"LR": 0.001}
    assert selection == {
        "method": "mean_validation_rmse",
        "seeds": [0, 1, 2],
        "mean_val_rmse": 1.0,
    }
    table["test_RMSE"] = [0, -1000, np.nan, 200, 300, 400]
    assert eb.select_configuration(table, [0, 1, 2]) == (config, selection)

    assert eb.select_configuration(table.drop(index=2), [0, 1, 2])[0] == {"LR": 0.01}
    with pytest.raises(ValueError, match="complete selection seed set"):
        eb.select_configuration(table.iloc[:2], [0, 1, 2])
    duplicate = table.iloc[:1].assign(val_RMSE=0.1)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        eb.select_configuration(pd.concat([table, duplicate]), [0, 1, 2])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("data_sha256", "old", "data_sha256"),
        ("split_sha256", "old", "split_sha256"),
        ("sweep_protocol", "old", "protocol"),
        ("experiment", "gnn_encoder", "experiment"),
        ("seed", -1, "training seeds"),
        ("val_RMSE", np.inf, "finite"),
        ("cfg", "[]", "JSON object"),
    ],
)
def test_load_sweep_records_normalizes_and_rejects_incompatible_files(
    bace1_data, tmp_path, column, value, message
):
    table = records(bace1_data, configs=[{"LR": 0.01, "EPOCHS": 10}])
    table["cfg"] = '{"LR": 0.01, "EPOCHS": 10}'
    path = tmp_path / "stage.csv"
    table.to_csv(path, index=False)
    loaded = eb.load_sweep_records(path, bace1_data, "ecfp4_encoder")
    assert loaded["cfg"].eq(bm.params_json({"EPOCHS": 10, "LR": 0.01})).all()

    table.loc[0, column] = value
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        eb.load_sweep_records(path, bace1_data, "ecfp4_encoder")


def test_run_cli_exploratory_resume_and_legacy_rejection(
    bace1_data, tmp_path, monkeypatch
):
    module = import_script("ECFP4_encoder_sweep")
    monkeypatch.setattr(eb.BACE1Benchmark, "load", lambda: bace1_data)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_csv = legacy / f"{next(iter(module.STAGES))}.csv"
    pd.DataFrame([measurements()]).to_csv(legacy_csv, index=False)
    before = legacy_csv.read_bytes()
    with pytest.raises(ValueError, match="legacy/incompatible.*preserve this file"):
        module.main(["1", "--output-dir", str(legacy)])
    assert legacy_csv.read_bytes() == before

    cfg = {"LR": 0.001}
    monkeypatch.setattr(module, "STAGES", {"1-tiny": ([cfg], [0])})
    calls = []

    def run(config, seed, cohort, *, evaluate_test=False):
        assert not evaluate_test
        calls.append((config, seed))
        return {
            **measurements(config, seed),
            **{f"test_{m}": np.nan for m in eb.METRICS},
        }

    monkeypatch.setattr(module, "run", run)
    out = tmp_path / "sweep"
    module.main(["1", "--output-dir", str(out)])
    path, log = out / "1-tiny.csv", out / "1-tiny.log"
    saved = (path.read_bytes(), log.read_bytes())
    module.main(["1", "--output-dir", str(out)])
    assert calls == [(cfg, 0)]
    assert (path.read_bytes(), log.read_bytes()) == saved
    assert pd.read_csv(path)["test_RMSE"].isna().all()


@pytest.mark.parametrize("script", SCRIPTS)
def test_best_epoch_restored_before_test_scoring(
    script, bace1_data, mocked_smiles_encoder, monkeypatch
):
    module = import_script(script)
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    base = {"EPOCHS": 2, "BATCH_SIZE": 4}
    if script == "SMILES_encoder_sweep":
        base["ENCODER_BATCH_SIZE"] = 3
    else:
        base |= {"HIDDEN_DIMS": [8], "DROPOUT": 0.0}
    if script == "ECFP4_encoder_sweep":
        base["N_BITS"] = 1024
    cfg = module.normalize_config(base)
    val_predictions = []

    def validation_metric(df, predictions):
        val_predictions.append(predictions["val"]["pIC50_BACE1"].clone())
        rmse = 0.1 if len(val_predictions) == 1 else 100.0
        return [{"metric": "rmse", "value": rmse}]

    monkeypatch.setattr(module, "evaluate_split_predictions", validation_metric)
    original_score = bace1_data.score

    def score_selected(predict, *, evaluate_test=False):
        assert len(val_predictions) == 2 and evaluate_test
        assert torch.equal(predict(bace1_data.indices["val"]), val_predictions[0])
        return original_score(predict, evaluate_test=evaluate_test)

    monkeypatch.setattr(bace1_data, "score", score_selected)
    trained = module.run(cfg, 42, bace1_data, evaluate_test=True)
    assert trained["best_epoch"] == 1
    assert np.isfinite(trained["test_RMSE"])


def test_gnn_training_batch_sampler_and_deterministic_context(bace1_data, monkeypatch):
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler

    module = import_script("gnn_encoder_sweep")
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))

    def batches(cls):
        gen = torch.Generator().manual_seed(42)
        sampler = cls(RandomSampler(range(60), generator=gen), 32, drop_last=False)
        return [b.tolist() for b in DataLoader(range(60), batch_sampler=sampler)]

    assert batches(module.TrainingBatchSampler) == batches(BatchSampler)
    tail_sampler = module.TrainingBatchSampler(range(8), 7, drop_last=False)
    assert list(tail_sampler) == [list(range(8))] and len(tail_sampler) == 1

    torch.use_deterministic_algorithms(False, warn_only=True)
    with pytest.raises(RuntimeError, match="boom"), module._deterministic_training():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise RuntimeError("boom")
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()


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
def test_smiles_requires_pinned_frozen_valid_configuration(bad):
    module = import_script("SMILES_encoder_sweep")
    with pytest.raises(ValueError):
        module.normalize_config(bad)


def test_smiles_head_training_preserves_order_frozen_weights_and_checks_revision(
    bace1_data, mocked_smiles_encoder, monkeypatch
):
    module = import_script("SMILES_encoder_sweep")
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        bace1_data,
        "features",
        lambda *a, **kw: pytest.fail("SMILES runs must not build fingerprints/graphs"),
    )
    cfg = module.normalize_config(
        {
            "EPOCHS": 2,
            "BATCH_SIZE": 4,
            "ENCODER_BATCH_SIZE": 3,
        }
    )

    val_only = module.run(cfg, 42, bace1_data, evaluate_test=False)
    assert np.isnan(val_only["test_RMSE"])
    test_smiles = set(bace1_data.frame.iloc[bace1_data.indices["test"]]["smiles"])
    assert not test_smiles.intersection(mocked_smiles_encoder.encoded)

    model = mocked_smiles_encoder.models[0]
    assert not model.training
    assert all(
        torch.equal(v, mocked_smiles_encoder.initial[0][k])
        for k, v in model.state_dict().items()
    )
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())

    full_run = module.run(cfg, 42, bace1_data, evaluate_test=True)
    changed = copy.deepcopy(bace1_data)
    changed.frame.loc[changed.indices["test"], "pIC50_BACE1"] += 1000
    changed_run = module.run(cfg, 42, changed, evaluate_test=True)
    assert full_run["val_RMSE"] == changed_run["val_RMSE"]
    assert full_run["test_RMSE"] != changed_run["test_RMSE"]

    original_make = module.make_encoder

    def wrong_revision(c):
        enc = original_make(c)
        enc.model.config._commit_hash = "b" * 40
        return enc

    monkeypatch.setattr(module, "make_encoder", wrong_revision)
    with pytest.raises(ValueError, match="revision does not match"):
        module.run(cfg, 42, bace1_data)
