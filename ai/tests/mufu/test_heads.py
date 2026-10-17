import pytest
import torch

from mufu.contract import FUSED_DIM, N_BIO, N_TOX, validate_outputs
from mufu.heads import (
    BioactivityClassificationHead,
    BioactivityRegressionHead,
    ToxicityHead,
)


@pytest.mark.parametrize(
    ("head_cls", "expected_width"),
    [
        (BioactivityRegressionHead, N_BIO),
        (BioactivityClassificationHead, N_BIO),
        (ToxicityHead, N_TOX),
    ],
)
def test_output_shape_and_gradients(head_cls, expected_width):
    batch_size = 4
    x = torch.randn(batch_size, FUSED_DIM, requires_grad=True)

    head = head_cls()
    output = head(x)

    assert output.shape == (batch_size, expected_width)
    output.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in head.parameters())


def test_heads_satisfy_mufu_outputs_contract():
    batch_size = 4
    x = torch.randn(batch_size, FUSED_DIM)

    outputs = {
        "pic50": BioactivityRegressionHead()(x),
        "active_logits": BioactivityClassificationHead()(x),
        "tox_logits": ToxicityHead()(x),
    }

    validate_outputs(outputs, batch_size)


@pytest.mark.parametrize(
    "head_cls",
    [BioactivityClassificationHead, ToxicityHead],
)
def test_classification_heads_return_unbounded_logits(head_cls):
    head = head_cls()
    x = torch.randn(2, FUSED_DIM)

    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.fill_(2.0)
    assert torch.all(head(x) > 1.0)

    with torch.no_grad():
        head.linear.bias.fill_(-2.0)
    assert torch.all(head(x) < 0.0)


def test_bioactivity_regression_head_configurable_dropout():
    head = BioactivityRegressionHead(dropout=0.25)

    dropouts = [m for m in head.modules() if isinstance(m, torch.nn.Dropout)]

    assert len(dropouts) == 1
    assert dropouts[0].p == 0.25


def test_bioactivity_regression_head_train_vs_eval_determinism():
    torch.manual_seed(0)
    head = BioactivityRegressionHead(dropout=0.5)
    x = torch.randn(8, FUSED_DIM)

    head.train()
    assert not torch.allclose(head(x), head(x))

    head.eval()
    assert torch.allclose(head(x), head(x))
