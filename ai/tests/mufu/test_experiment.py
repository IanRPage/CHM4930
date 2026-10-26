import json
import random

import pandas as pd
import torch
from conftest import SmallSmilesEncoder

from baseline import results
from mufu import experiment
from mufu.contract import D_MODEL, MODALITY_DROPOUT_P, N_BIO
from mufu.mc_dropout import load_conformal
from mufu.train import TrainConfig


def test_benchmark_params_drops_runtime_fields_and_adds_model_settings():
    cfg = TrainConfig(lr=5e-4, device="cuda", num_workers=4)

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
    cfg = TrainConfig(
        batch_size=4, epochs=2, patience=2, gnn_hidden_dims=(32, 16), device="cpu"
    )
    encoder = SmallSmilesEncoder().requires_grad_(False)
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
    assert not any(k.startswith("smiles_encoder.") for k in saved["model"])
    conformal = load_conformal(tmp_path / "checkpoints" / "mufu_test_seed_3.pt")
    assert conformal.q.shape == (N_BIO, len(conformal.coverages))
    coverage = pd.read_csv(tmp_path / "outputs" / "seed_3_mc_dropout_coverage.csv")
    assert list(coverage.columns) == [
        "target",
        "nominal",
        "empirical",
        "n",
        "mean_width",
    ]
    history = pd.read_csv(tmp_path / "outputs" / "seed_3_loss.csv")
    assert list(history.columns) == [
        "epoch",
        "train_loss",
        "val_loss",
        "val_pic50_loss",
        "val_bio_cls_loss",
        "val_tox_loss",
    ]
    assert len(history) == 2

    loaded_model, meta = experiment.load_mufu_checkpoint(
        tmp_path / "checkpoints" / "mufu_test_seed_3.pt", encoder
    )
    re_preds = experiment.predict(
        loaded_model,
        labeled_dataset,
        batch_size=cfg.batch_size,
        family_states=meta["family_states"],
        calibration=meta["calibration"],
    )
    for col in preds["test"]:
        if col == "smiles":
            assert re_preds[col] == preds["test"][col]
        else:
            assert (re_preds[col] == preds["test"][col]).all()
