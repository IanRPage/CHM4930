"""
MLP encoder mapping fixed-width ECFP4 fingerprint bit vectors into molecular embeddings.
"""

from collections.abc import Sequence

import torch
from torch import nn

from featurize.fingerprint import FINGERPRINT_BITS


class FingerprintEncoder(nn.Module):
    """
    MLP encoder for ECFP4 fingerprints

    Maps a batch of fixed-size bit vectors to a pooled vector, and `forward`
    returns the last hidden layer (width `out_dim`).
    """

    def __init__(
        self,
        in_bits: int = FINGERPRINT_BITS,
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.2,
        final_activation: bool = True,
    ) -> None:
        super().__init__()
        self.in_bits = in_bits
        self.out_dim = hidden_dims[-1]

        layers: list[nn.Module] = []
        width = in_bits
        for i, hidden in enumerate(hidden_dims):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden)]
            if final_activation or i < len(hidden_dims) - 1:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.net = nn.Sequential(*layers)

    # x is [batch, in_bits]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_bits:
            raise ValueError(
                f"expected input of shape (batch, {self.in_bits}), got {tuple(x.shape)}"
            )
        return self.net(x.float())
