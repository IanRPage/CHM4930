import torch

from mufu.contract import FUSED_DIM, N_TOX
from mufu.toxicity_head import ToxicityHead


def test_output_shape():
    batch_size = 4
    x = torch.randn(batch_size, FUSED_DIM)

    toxicity_head = ToxicityHead()
    output = toxicity_head(x)

    assert output.shape == (batch_size, N_TOX)


def test_returns_logits_without_sigmoid():
    toxicity_head = ToxicityHead()

    with torch.no_grad():
        toxicity_head.linear.weight.zero_()
        toxicity_head.linear.bias.fill_(2.0)

    x = torch.randn(2, FUSED_DIM)
    output = toxicity_head(x)

    assert torch.all(output > 1.0)
