import pytest
import torch

from mufu.contract import DROPOUT_MASKS, MODALITIES
from mufu.modality_dropout import sample_modality_mask


def test_disabled_sampling_keeps_all_modalities_without_advancing_rng():
    generator = torch.Generator().manual_seed(42)
    state = generator.get_state()

    mask = sample_modality_mask(8, p=0.0, generator=generator)

    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.ones(8, len(MODALITIES), dtype=torch.bool))
    assert torch.equal(generator.get_state(), state)


def test_probability_one_only_samples_nonempty_partial_patterns():
    mask = sample_modality_mask(
        128, p=1.0, device="cpu", generator=torch.Generator().manual_seed(42)
    )

    assert mask.dtype == torch.bool
    assert set(map(tuple, mask.tolist())) == set(DROPOUT_MASKS)


def test_dropout_probability_is_per_sample_not_per_modality():
    size = 40_000
    mask = sample_modality_mask(
        size, p=0.3, generator=torch.Generator().manual_seed(17)
    )
    codes = mask.long() @ torch.tensor([4, 2, 1])
    counts = torch.bincount(codes, minlength=8)
    expected = torch.tensor([0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7])

    assert counts[0] == 0
    assert torch.allclose(counts / size, expected, atol=0.015, rtol=0)


def test_generator_replays_sequence_but_advances_between_calls():
    generator = torch.Generator().manual_seed(42)
    first = sample_modality_mask(64, p=0.3, generator=generator)
    second = sample_modality_mask(64, p=0.3, generator=generator)

    replay = torch.Generator().manual_seed(42)
    assert torch.equal(first, sample_modality_mask(64, p=0.3, generator=replay))
    assert torch.equal(second, sample_modality_mask(64, p=0.3, generator=replay))
    assert not torch.equal(first, second)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        sample_modality_mask(4, p=p)


def test_rejects_negative_batch_size():
    with pytest.raises(ValueError):
        sample_modality_mask(-1)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="needs MPS"
            ),
        ),
    ],
)
def test_sampling_on_accelerator(device):
    mask = sample_modality_mask(64, p=0.3, device=device)

    assert mask.device.type == device
    assert mask.dtype == torch.bool
    assert mask.any(dim=1).all()
