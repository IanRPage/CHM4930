import math

import pytest
import torch

from mufu import contract as c
from mufu.uncertainty_weighting import UncertaintyWeighting


def make_task_losses(pic50=2.0, bio_cls=0.6, tox=0.8):
    return {
        "pic50": None if pic50 is None else torch.tensor(float(pic50)),
        "bio_cls": None if bio_cls is None else torch.tensor(float(bio_cls)),
        "tox": None if tox is None else torch.tensor(float(tox)),
    }


def test_default_initialization():
    weighting = UncertaintyWeighting()

    assert set(weighting.log_vars.keys()) == set(c.TASKS)
    for task in c.TASKS:
        param = weighting.log_vars[task]
        assert param.shape == ()
        assert param.requires_grad
        assert torch.isclose(param, torch.tensor(c.LOG_VAR_INIT))


def test_forward_weights_and_regularizes_tasks():
    weighting = UncertaintyWeighting()
    s_values = {"pic50": 0.4, "bio_cls": -0.3, "tox": 1.1}
    raw_losses = {"pic50": 2.5, "bio_cls": 0.7, "tox": 1.2}

    with torch.no_grad():
        for task, val in s_values.items():
            weighting.log_vars[task].fill_(val)

    total = weighting(make_task_losses(**raw_losses))

    expected = sum(
        c.TASK_SPECS[task].loss_scale * math.exp(-s_values[task]) * raw_losses[task]
        + 0.5 * s_values[task]
        for task in c.TASKS
    )
    assert total.shape == ()
    assert torch.isclose(total, torch.tensor(expected, dtype=torch.float32))


def test_skips_none_tasks_including_regularization_term():
    weighting = UncertaintyWeighting()
    with torch.no_grad():
        weighting.log_vars["pic50"].fill_(0.5)
        weighting.log_vars["bio_cls"].fill_(-0.2)
        weighting.log_vars["tox"].fill_(3.0)

    pic50_loss = torch.tensor(2.0, requires_grad=True)
    bio_cls_loss = torch.tensor(0.8, requires_grad=True)
    total = weighting({"pic50": pic50_loss, "bio_cls": bio_cls_loss, "tox": None})

    expected = (
        0.5 * math.exp(-0.5) * 2.0
        + 0.5 * 0.5
        + 1.0 * math.exp(0.2) * 0.8
        + 0.5 * (-0.2)
    )
    assert torch.isclose(total, torch.tensor(expected, dtype=torch.float32))

    total.backward()
    assert weighting.log_vars["pic50"].grad is not None
    assert weighting.log_vars["bio_cls"].grad is not None
    assert weighting.log_vars["tox"].grad is None
    assert pic50_loss.grad is not None
    assert bio_cls_loss.grad is not None


def test_all_none_returns_zero_scalar_with_grad():
    weighting = UncertaintyWeighting(init_log_var=1.5)
    total = weighting({"pic50": None, "bio_cls": None, "tox": None})

    assert total.shape == ()
    assert total.requires_grad
    assert torch.isclose(total, torch.tensor(0.0))

    total.backward()
    for task in c.TASKS:
        assert torch.isclose(weighting.log_vars[task].grad, torch.tensor(0.0))


def test_rejects_nan_loss():
    weighting = UncertaintyWeighting()
    losses = {**make_task_losses(), "tox": torch.tensor(float("nan"))}
    with pytest.raises(ValueError, match="tox: loss is NaN"):
        weighting(losses)


@pytest.mark.parametrize("bad_val", [float("inf"), float("-inf")])
def test_rejects_inf_loss(bad_val):
    weighting = UncertaintyWeighting()
    losses = {**make_task_losses(), "pic50": torch.tensor(bad_val)}
    with pytest.raises(ValueError, match="pic50: loss is NaN or Inf"):
        weighting(losses)


def test_rejects_negative_loss():
    weighting = UncertaintyWeighting()
    losses = {**make_task_losses(), "bio_cls": torch.tensor(-0.1)}
    with pytest.raises(ValueError, match="bio_cls: loss must be non-negative"):
        weighting(losses)


def test_clamps_extreme_log_vars():
    weighting = UncertaintyWeighting()
    with torch.no_grad():
        weighting.log_vars["pic50"].fill_(-100.0)
        weighting.log_vars["bio_cls"].fill_(100.0)

    total = weighting(
        {"pic50": torch.tensor(1.0), "bio_cls": torch.tensor(1.0), "tox": None}
    )
    expected = (
        0.5 * math.exp(5.0) * 1.0
        + 0.5 * (-5.0)
        + 1.0 * math.exp(-5.0) * 1.0
        + 0.5 * 5.0
    )
    assert torch.isfinite(total)
    assert torch.isclose(total, torch.tensor(expected, dtype=torch.float32))
