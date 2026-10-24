import json
import random

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import SmallSmilesEncoder

from baseline import results
from mufu import experiment
from mufu.contract import D_MODEL, MODALITY_DROPOUT_P
from mufu.train import TrainConfig


def test_benchmark_params_drops_runtime_fields_and_adds_model_settings():
    cfg = TrainConfig(lr=5e-4, device="cuda", num_workers=4, checkpoint_path="x.pt")

    params = experiment.benchmark_params(cfg, SmallSmilesEncoder())

    assert not set(experiment.RUNTIME_FIELDS) & set(params)
    assert params["lr"] == 5e-4
    assert params["min_delta"] == cfg.min_delta
    assert params["d_model"] == D_MODEL
    assert params["modality_dropout_p"] == MODALITY_DROPOUT_P
    assert params["smiles_encoder"] == "SmallSmilesEncoder"
    json.dumps(params)


def test_fit_predict_saves_checkpoint_metadata_and_loss_history(
    labeled_dataset, tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment, "git_commit", lambda: "abc1234")
    data_path = tmp_path / "combined.csv"
    data_path.write_text("smiles\nCCO\n")
    datasets = {split: labeled_dataset for split in ("train", "val", "test")}
    cfg = TrainConfig(batch_size=4, epochs=2, patience=2, device="cpu")
    encoder = SmallSmilesEncoder()
    fit_predict = experiment.make_fit_predict(
        datasets,
        encoder,
        cfg,
        model_name="mufu_test",
        checkpoint_dir=tmp_path / "checkpoints",
        history_dir=tmp_path / "outputs",
        data_path=data_path,
    )

    torch.manual_seed(0)
    random.seed(0)
    preds = fit_predict(pd.DataFrame(), 3)

    assert set(preds) == {"val", "test"}
    for split_preds in preds.values():
        assert split_preds["smiles"] == labeled_dataset.smiles
    saved = torch.load(
        tmp_path / "checkpoints" / "mufu_test_seed_3.pt", weights_only=True
    )
    assert saved["seed"] == 3
    assert saved["params"] == experiment.benchmark_params(cfg, encoder)
    assert saved["git_commit"] == "abc1234"
    assert saved["data_sha256"] == results.file_sha256(data_path)
    assert 1 <= saved["epoch"] <= 2
    assert all(t.device.type == "cpu" for t in saved["model"].values())
    history = pd.read_csv(tmp_path / "outputs" / "seed_3_loss.csv")
    assert list(history.columns) == ["epoch", "train_loss", "val_loss"]
    assert len(history) == 2


def test_run_mufu_benchmark_records_finished_seeds_before_a_crash(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "git_commit", lambda: "abc1234")
    path = tmp_path / "results.csv"
    data_path = tmp_path / "combined.csv"
    data_path.write_text("smiles\nCCO\n")
    df = pd.DataFrame(
        {
            "smiles": ["C", "CC", "CCC", "CCCC"],
            "split": ["val", "val", "test", "test"],
            "pIC50_BACE1": [5.0, 7.0, 4.0, 8.0],
            "active_BACE1": [0.0, 1.0, 0.0, 1.0],
        }
    )

    def fit_predict(df, seed):
        if seed == 1:
            raise RuntimeError("crash")
        return {
            "val": {"pIC50_BACE1": np.array([5.0, 7.0])},
            "test": {"pIC50_BACE1": np.array([4.0, 8.0])},
        }

    with pytest.raises(RuntimeError, match="crash"):
        experiment.run_mufu_benchmark(
            fit_predict,
            model_name="mufu",
            encoder="enc",
            params={"lr": 1e-3},
            seeds=(0, 1),
            df=df,
            path=path,
            data_path=data_path,
        )

    written = pd.read_csv(path)
    assert set(written["seed"]) == {0}
    assert set(written["model"]) == {"mufu"}
