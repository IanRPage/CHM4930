import copy
import random

import numpy as np
import pytest
import torch

from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TASKS, TOX_COLUMNS
from mufu.data.dataset import collate
from mufu.loss.task_loss import TaskLoss, compute_task_losses
from mufu.loss.uncertainty_weighting import UncertaintyWeighting
from mufu.train import (
    TrainConfig,
    cpu_state,
    fit,
    load_non_smiles_state,
    make_optimizer,
    predict,
    train_step,
)


def scripted_losses(total):
    return None, {"pic50": total, "bio_cls": 0.0, "tox": 0.0}


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


def test_fit_restores_best_val_state_and_early_stops(
    model, labeled_dataset, monkeypatch
):
    torch.manual_seed(0)
    random.seed(0)

    scripted_val_losses = iter([2.0, 1.0, 1.5, 1.8, 2.1])
    epoch_snapshots = []

    def fake_evaluate_val_loss(m, val_dataset, task_loss, batch_size):
        epoch_snapshots.append(copy.deepcopy(m.state_dict()))
        return scripted_losses(next(scripted_val_losses))

    monkeypatch.setattr("mufu.train.evaluate_val_loss", fake_evaluate_val_loss)
    cfg = TrainConfig(
        lr=5e-3,
        batch_size=4,
        epochs=10,
        patience=2,
        sampler="alternating",
        device="cpu",
    )

    result = fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert result.best_epoch == 2
    assert result.best_val_loss == 1.0
    assert result.val_losses == [2.0, 1.0, 1.5, 1.8]
    for key, tensor in epoch_snapshots[1].items():
        if not key.startswith("smiles_encoder."):
            assert torch.equal(model.state_dict()[key], tensor)


def test_fit_stops_on_val_loss_without_pos_weight(model, labeled_dataset, monkeypatch):
    val_losses_used = []

    def fake_evaluate_val_loss(m, val_dataset, task_loss, batch_size):
        val_losses_used.append(task_loss)
        return scripted_losses(1.0)

    monkeypatch.setattr("mufu.train.evaluate_val_loss", fake_evaluate_val_loss)
    cfg = TrainConfig(batch_size=4, epochs=2, patience=2, device="cpu")

    fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert TaskLoss.from_dataset(labeled_dataset).tox_pos_weight is not None
    for task_loss in val_losses_used:
        assert task_loss.bio_pos_weight is None
        assert task_loss.tox_pos_weight is None


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
        "mufu.train.evaluate_val_loss",
        lambda *args: scripted_losses(next(scripted_val_losses)),
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


def test_cpu_state_excludes_smiles_encoder_and_load_non_smiles_state_validates(model):
    snap = cpu_state(model)
    assert not any(k.startswith("smiles_encoder.") for k in snap)
    assert any(k.startswith("fingerprint_encoder.") for k in snap)

    with torch.no_grad():
        for p in model.fingerprint_encoder.parameters():
            p.add_(5.0)
    load_non_smiles_state(model, snap)
    for k, v in snap.items():
        assert torch.equal(model.state_dict()[k], v)

    incomplete = {k: v for k, v in snap.items() if not k.startswith("fusion.")}
    with pytest.raises(ValueError, match="missing="):
        load_non_smiles_state(model, incomplete)

    unexpected = {**snap, "bogus.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="unexpected="):
        load_non_smiles_state(model, unexpected)


def test_fit_uses_load_non_smiles_state_to_restore_best_model(
    model, labeled_dataset, monkeypatch
):
    calls = []
    real_load = load_non_smiles_state

    def spy_load(m, state):
        calls.append(set(state))
        return real_load(m, state)

    monkeypatch.setattr("mufu.train.load_non_smiles_state", spy_load)
    cfg = TrainConfig(batch_size=4, epochs=2, patience=2, device="cpu")
    fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert len(calls) == 1
    assert not any(k.startswith("smiles_encoder.") for k in calls[0])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sampler": "bad"}, "unknown sampler"),
        ({"checkpoint_mode": "bad"}, "unknown checkpoint_mode"),
        ({"pos_weight": "bad"}, "unknown pos_weight"),
        ({"stopping": "bad"}, "unknown stopping"),
        (
            {"stopping": "metric", "checkpoint_mode": "single"},
            "requires checkpoint_mode='task_family'",
        ),
        ({"calibration": "bad"}, "unknown calibration"),
        ({"task_weights": (1.0, 2.0)}, "task_weights: expected 3"),
    ],
)
def test_fit_rejects_invalid_config_options(model, labeled_dataset, kwargs, match):
    cfg = TrainConfig(batch_size=4, epochs=1, device="cpu", **kwargs)
    with pytest.raises(ValueError, match=match):
        fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)


def test_train_step_returns_unweighted_loss_and_supports_fixed_task_weights(
    model, labeled_dataset
):
    torch.manual_seed(0)
    model.eval()
    batch = collate([labeled_dataset[i] for i in range(len(labeled_dataset))])
    weighted_loss_fn = TaskLoss(
        bio_pos_weight=torch.full((2,), 10.0),
        tox_pos_weight=torch.full((14,), 25.0),
    )
    weighting = UncertaintyWeighting()
    optimizer = make_optimizer(model, weighting, TrainConfig(lr=1e-3, log_var_lr=5e-2))
    assert optimizer.param_groups[1]["lr"] == 5e-2

    with torch.no_grad():
        expected_unweighted = sum(
            v
            for v in compute_task_losses(model(batch), batch).values()
            if v is not None
        )
        weighted_sum = sum(
            v for v in weighted_loss_fn(model(batch), batch).values() if v is not None
        )
    assert not torch.isclose(expected_unweighted, weighted_sum)

    logged = train_step(
        model,
        batch,
        weighted_loss_fn,
        weighting,
        optimizer,
        task_weights=(1.0, 0.5, 0.25),
    )
    assert torch.isclose(logged, expected_unweighted, atol=1e-5)
    for p in weighting.log_vars.values():
        assert p.grad is None


def test_task_family_checkpointing_and_non_mutating_predict(model, labeled_dataset):
    torch.manual_seed(0)
    random.seed(0)
    cfg = TrainConfig(
        batch_size=4,
        epochs=4,
        patience=2,
        checkpoint_mode="task_family",
        stopping="metric",
        device="cpu",
    )

    result = fit(model, {"train": labeled_dataset, "val": labeled_dataset}, cfg)

    assert set(result.val_task_losses) == set(TASKS)
    assert all(
        len(v) == len(result.val_losses) for v in result.val_task_losses.values()
    )
    assert result.best_epochs is not None
    assert set(result.best_epochs) == {"bio", "tox"}
    assert result.best_epoch == result.best_epochs["bio"]
    assert result.family_states is not None
    assert set(result.family_states) == {"tox"}
    assert not any(k.startswith("smiles_encoder.") for k in result.family_states["tox"])

    modified_tox = copy.deepcopy(result.family_states["tox"])
    modified_tox["toxicity_head.linear.bias"].fill_(50.0)
    before_state = copy.deepcopy(model.state_dict())

    preds = predict(
        model,
        labeled_dataset,
        batch_size=4,
        family_states={"tox": modified_tox},
    )

    for k, v in before_state.items():
        assert torch.equal(model.state_dict()[k], v), k
    for col in TOX_COLUMNS:
        assert np.allclose(preds[col], 1.0)
