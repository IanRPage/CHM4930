import torch

from mufu.bioactivity_regression_head import BioactivityRegressionHead
from mufu.contract import FUSED_DIM, N_BIO


def test_output_shape():
    batch_size = 4
    x = torch.randn(batch_size, FUSED_DIM)

    bioactivity_regression_head = BioactivityRegressionHead()
    output = bioactivity_regression_head(x)

    assert output.shape == (batch_size, N_BIO)


def test_contains_configurable_dropout():
    bioactivity_regression_head = BioactivityRegressionHead(dropout=0.25)

    dropouts = [
        module
        for module in bioactivity_regression_head.modules()
        if isinstance(module, torch.nn.Dropout)
    ]

    assert len(dropouts) >= 1
    assert dropouts[0].p == 0.25
