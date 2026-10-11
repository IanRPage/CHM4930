import torch

from mufu.bioactivity_classification_head import BioactivityClassificationHead
from mufu.contract import N_BIO


def test_output_shape():
    batch_size = 4
    x = torch.randn(batch_size, N_BIO)

    bioactivity_classification_head = BioactivityClassificationHead()
    output = bioactivity_classification_head(x)

    assert output.shape == (batch_size, N_BIO)


def test_returns_logits_without_sigmoid():
    bioactivity_classification_head = BioactivityClassificationHead()

    with torch.no_grad():
        bioactivity_classification_head.linear.weight.zero_()
        bioactivity_classification_head.linear.bias.fill_(2.0)

    x = torch.randn(2, N_BIO)
    output = bioactivity_classification_head(x)

    assert torch.all(output > 1.0)
