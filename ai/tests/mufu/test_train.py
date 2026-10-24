import copy
import random

import numpy as np
import torch

from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import collate
from mufu.loss.task_loss import TaskLoss
from mufu.loss.uncertainty_weighting import UncertaintyWeighting
from mufu.train import TrainConfig, fit, make_optimizer, predict, train_step


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
    model, labeled_dataset, dataset
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

    unlabeled_batch = collate([dataset[i] for i in range(len(dataset))])
    skipped = train_step(model, unlabeled_batch, task_loss, weighting, optimizer)

    assert skipped is None
    for name, param in [*model.named_parameters(), *weighting.named_parameters()]:
        assert torch.equal(param, params_before[name]), name


def test_predict_output_keys_shapes_and_probabilities(model, dataset):
    with torch.no_grad():
        model.classification_head.linear.bias.fill_(10.0)
        model.toxicity_head.linear.bias.fill_(-10.0)
    model.train()

    preds = predict(model, dataset, batch_size=3)

    assert not model.training
    expected_cols = (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS)
    assert tuple(preds) == ("smiles", *expected_cols)
    assert preds["smiles"] == dataset.smiles
    for col in expected_cols:
        assert isinstance(preds[col], np.ndarray)
        assert preds[col].shape == (len(dataset),)
        assert preds[col].dtype == float
    for col in (*ACTIVE_COLUMNS, *TOX_COLUMNS):
        assert ((0.0 <= preds[col]) & (preds[col] <= 1.0)).all()


def test_fit_min_delta_ignores_tiny_improvements(model, labeled_dataset, monkeypatch):
    torch.manual_seed(0)
    random.seed(0)
    scripted_val_losses = iter([2.0, 1.9995, 1.9993, 1.9991, 1.9])
    monkeypatch.setattr(
        "mufu.train.evaluate_val_loss", lambda *args: next(scripted_val_losses)
    )
    cfg = TrainConfig(batch_size=4, epochs=5, patience=3, min_delta=1e-3, device="cpu")

    result = fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert result.best_epoch == 1
    assert result.best_val_loss == 2.0
    assert len(result.val_losses) == 4


def test_fit_verbose_prints_one_line_per_epoch(model, labeled_dataset, capsys):
    torch.manual_seed(0)
    random.seed(0)
    cfg = TrainConfig(batch_size=4, epochs=3, patience=3, device="cpu")

    fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg, verbose=True)

    lines = capsys.readouterr().out.splitlines()
    assert [line.split(" |")[0] for line in lines] == [
        "Epoch 1/3",
        "Epoch 2/3",
        "Epoch 3/3",
    ]
