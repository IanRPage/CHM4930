import copy
import random

import numpy as np
import pytest
import torch
from test_model import SMILES, SmallSmilesEncoder, unlabeled_frame

from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.dataset import MuFuDataset, collate
from mufu.model import MuFu
from mufu.task_loss import TaskLoss
from mufu.train import TrainConfig, fit, make_optimizer, predict, train_step
from mufu.uncertainty_weighting import UncertaintyWeighting


@pytest.fixture
def labeled_dataset():
    frame = unlabeled_frame()
    bio_rows, tox_rows = frame.index[:4], frame.index[4:]
    for j, column in enumerate(PIC50_COLUMNS):
        frame.loc[bio_rows, column] = [5.0 + j + 0.5 * i for i in range(4)]
    for j, column in enumerate(ACTIVE_COLUMNS):
        frame.loc[bio_rows, column] = [float((i + j) % 2) for i in range(4)]
    for j, column in enumerate(TOX_COLUMNS):
        frame.loc[tox_rows, column] = [float((i + j) % 2) for i in range(4)]
    frame.loc[1, [PIC50_COLUMNS[-1], ACTIVE_COLUMNS[-1]]] = float("nan")
    return MuFuDataset(frame)


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return MuFu(
            smiles_encoder=SmallSmilesEncoder(),
            fingerprint_encoder=FingerprintEncoder(hidden_dims=(16,), dropout=0.0),
            graph_encoder=GNNEncoder(hidden_dims=(12,), dropout=0.0),
        )


def test_fit_decreases_loss_on_tiny_dataset(model, labeled_dataset):
    torch.manual_seed(0)
    random.seed(0)
    cfg = TrainConfig(
        lr=3e-3,
        batch_size=4,
        epochs=20,
        patience=20,
        device="cpu",
    )

    result = fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert len(result.train_losses) == 20
    assert len(result.val_losses) == 20
    assert result.train_losses[-1] < result.train_losses[0]
    assert result.val_losses[-1] < result.val_losses[0]
    assert result.best_val_loss == min(result.val_losses)


def test_fit_restores_best_val_checkpoint_and_early_stops(
    model, labeled_dataset, tmp_path, monkeypatch
):
    torch.manual_seed(0)
    random.seed(0)

    scripted_val_losses = iter([2.0, 1.0, 1.5, 1.8, 2.1])
    epoch_snapshots = []

    def fake_evaluate_val_loss(m, val_dataset, task_loss, batch_size):
        epoch_snapshots.append(copy.deepcopy(m.state_dict()))
        return next(scripted_val_losses)

    monkeypatch.setattr("mufu.train.evaluate_val_loss", fake_evaluate_val_loss)
    ckpt_path = tmp_path / "checkpoints" / "best.pt"
    cfg = TrainConfig(
        lr=5e-3,
        batch_size=4,
        epochs=10,
        patience=2,
        sampler="alternating",
        checkpoint_path=ckpt_path,
        device="cpu",
    )

    result = fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert result.best_epoch == 2
    assert result.best_val_loss == 1.0
    assert result.val_losses == [2.0, 1.0, 1.5, 1.8]
    assert ckpt_path.exists()

    saved = torch.load(ckpt_path, weights_only=True)
    assert set(saved) == {"model", "weighting", "epoch"}
    assert saved["epoch"] == 2
    for key, tensor in epoch_snapshots[1].items():
        assert torch.equal(model.state_dict()[key], tensor)
        assert torch.equal(saved["model"][key], tensor)
    for key, tensor in saved["weighting"].items():
        assert torch.equal(result.weighting.state_dict()[key], tensor)


def test_make_optimizer_excludes_log_vars_from_weight_decay(model):
    weighting = UncertaintyWeighting()
    cfg = TrainConfig(lr=2e-3, weight_decay=0.05)

    optimizer = make_optimizer(model, weighting, cfg)

    assert len(optimizer.param_groups) == 2
    model_group, log_vars_group = optimizer.param_groups
    assert model_group["weight_decay"] == cfg.weight_decay
    assert model_group["lr"] == cfg.lr
    assert log_vars_group["weight_decay"] == 0.0
    assert log_vars_group["lr"] == cfg.lr
    assert [id(p) for p in log_vars_group["params"]] == [
        id(p) for p in weighting.log_vars.values()
    ]
    assert set(map(id, model_group["params"])).isdisjoint(
        map(id, log_vars_group["params"])
    )


def test_train_step_skips_unlabeled_batch_leaving_all_params_unchanged(
    model, labeled_dataset
):
    torch.manual_seed(0)
    task_loss = TaskLoss.from_dataset(labeled_dataset)
    weighting = UncertaintyWeighting(init_log_var=1.2)
    optimizer = make_optimizer(model, weighting, TrainConfig(lr=1e-2, weight_decay=0.1))
    model.train()
    weighting.train()

    labeled_batch = collate([labeled_dataset[i] for i in range(len(labeled_dataset))])
    first_loss = train_step(model, labeled_batch, task_loss, weighting, optimizer)
    assert first_loss is not None
    assert len(optimizer.state) > 0

    params_before = {
        name: param.detach().clone()
        for name, param in [
            *model.named_parameters(),
            *weighting.named_parameters(),
        ]
    }

    unlabeled_ds = MuFuDataset(unlabeled_frame())
    unlabeled_batch = collate([unlabeled_ds[i] for i in range(len(unlabeled_ds))])
    skipped = train_step(model, unlabeled_batch, task_loss, weighting, optimizer)

    assert skipped is None
    for name, param in [*model.named_parameters(), *weighting.named_parameters()]:
        assert torch.equal(param, params_before[name]), name


def test_predict_output_keys_and_shapes(model):
    ds = MuFuDataset(unlabeled_frame())
    model.train()

    preds = predict(model, ds, batch_size=3)

    assert not model.training
    expected_keys = (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS)
    assert tuple(preds) == expected_keys
    for col in expected_keys:
        assert isinstance(preds[col], np.ndarray)
        assert preds[col].shape == (len(SMILES),)
        assert preds[col].dtype == float
