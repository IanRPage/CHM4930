from collections.abc import Mapping

import torch
from torch import nn

from mufu.contract import D_MODEL, FUSED_DIM, MODALITIES, _check_keys, _check_tensor


class LateFusion(nn.Module):
    """
    Late fusion of the projected modality vectors into the shared latent L.

    `mask` is an optional (B, len(MODALITIES)) bool tensor whose column i gates
    MODALITIES[i], where None means all present.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.LayerNorm(FUSED_DIM),
            nn.GELU(),
        )

    def forward(
        self,
        projected: Mapping[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_keys("projected", set(projected), set(MODALITIES))
        size = len(projected[MODALITIES[0]])
        for m in MODALITIES:
            _check_tensor(m, projected[m], (size, D_MODEL), "float")
        if mask is None:
            fused = torch.cat([projected[m] for m in MODALITIES], dim=1)
        else:
            _check_tensor("mask", mask, (size, len(MODALITIES)), "bool")
            mask = mask.to(device=projected[MODALITIES[0]].device)
            fused = torch.cat(
                [projected[m] * mask[:, i : i + 1] for i, m in enumerate(MODALITIES)],
                dim=1,
            )
        return self.block(fused)
