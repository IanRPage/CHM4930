from collections.abc import Sequence

import torch
from torch import nn


class FingerprintEncoder(nn.Module):
    """
    MLP encoder for ECFP4 fingerprints

    Maps a batch of fixed-size bit vectors to a pooled vector, and `forward`
    returns the last hidden layer (width `out_dim`).
    """

    def __init__(
        self,
        in_bits: int = 2048,
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_bits = in_bits
        self.out_dim = hidden_dims[-1]

        # each hidden layer is Linear -> LayerNorm -> ReLU -> Dropout
        layers: list[nn.Module] = []
        width = in_bits
        for hidden in hidden_dims:
            layers += [
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width = hidden
        self.net = nn.Sequential(*layers)

    # x is [batch, in_bits]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_bits:
            raise ValueError(
                f"expected input of shape (batch, {self.in_bits}), got {tuple(x.shape)}"
            )
        return self.net(x.float())
