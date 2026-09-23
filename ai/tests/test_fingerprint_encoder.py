import pytest
import torch
from torch import nn

from fingerprint_encoder import FingerprintEncoder


def bits(batch=4, in_bits=2048):
    return torch.randint(0, 2, (batch, in_bits), dtype=torch.uint8)


@pytest.mark.parametrize(
    ("in_bits", "hidden_dims"),
    [(1024, (1024, 512)), (2048, (1024, 512)), (2048, (256,)), (1024, (512, 256, 128))],
)
def test_output_shape(in_bits, hidden_dims):
    enc = FingerprintEncoder(in_bits, hidden_dims)
    out = enc(bits(3, in_bits))
    assert enc.out_dim == hidden_dims[-1]
    assert out.shape == (3, hidden_dims[-1])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bool, torch.float32])
def test_accepts_bit_dtypes_and_returns_float32(dtype):
    out = FingerprintEncoder()(bits().to(dtype))
    assert out.dtype == torch.float32


@pytest.mark.parametrize("shape", [(2048,), (4, 1024), (2, 4, 2048)])
def test_rejects_wrong_shape(shape):
    with pytest.raises(ValueError, match="expected input of shape"):
        FingerprintEncoder(2048)(torch.zeros(shape))


def test_eval_is_deterministic():
    enc = FingerprintEncoder(dropout=0.5).eval()
    x = bits()
    assert torch.equal(enc(x), enc(x))


def test_train_mode_dropout_is_stochastic():
    enc = FingerprintEncoder(dropout=0.5).train()
    x = bits()
    assert not torch.equal(enc(x), enc(x))


def test_zero_dropout_is_deterministic_in_train_mode():
    enc = FingerprintEncoder(dropout=0.0).train()
    x = bits()
    assert torch.equal(enc(x), enc(x))


def test_gradients_reach_every_parameter():
    enc = FingerprintEncoder()
    enc(bits()).sum().backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_all_zero_input_gives_finite_output():
    # a dropped modality is zeroed out before it reaches the encoder
    out = FingerprintEncoder().eval()(torch.zeros(2, 2048))
    assert torch.isfinite(out).all()


def test_uses_layer_norm_and_dropout_per_hidden_layer():
    enc = FingerprintEncoder(hidden_dims=(64, 32, 16))
    kinds = [type(m) for m in enc.modules()]
    assert kinds.count(nn.LayerNorm) == 3
    assert kinds.count(nn.Dropout) == 3
    assert not any(isinstance(m, nn.BatchNorm1d) for m in enc.modules())
