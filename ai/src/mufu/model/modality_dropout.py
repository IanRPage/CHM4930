"""
Modality presence mask sampling for training-time input modality dropout.
"""

import torch

from mufu.contract import DROPOUT_MASKS, MODALITIES, MODALITY_DROPOUT_P


def sample_modality_mask(
    batch_size: int,
    *,
    p: float = MODALITY_DROPOUT_P,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample a (B, len(MODALITIES)) bool presence mask in MODALITIES order.

    Each sample stays fully present with probability 1-p; otherwise choose
    uniformly from DROPOUT_MASKS. At p=0.3 this gives 70% full presence and
    5% for each of the six partial patterns. Every row retains a modality.

    Call during training only; pass mask=None to fusion for normal evaluation.
    p=0 does not consume randomness.
    """
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    # Turn off modality dropout
    if p == 0.0:
        return torch.ones(batch_size, len(MODALITIES), dtype=torch.bool, device=device)

    patterns = torch.tensor(DROPOUT_MASKS, dtype=torch.bool, device=device)
    indices = torch.randint(
        len(DROPOUT_MASKS), (batch_size,), device=device, generator=generator
    )
    mask = patterns[indices]
    if p < 1.0:
        keep_all = torch.rand(batch_size, device=device, generator=generator) >= p
        mask.masked_fill_(keep_all[:, None], True)
    return mask
