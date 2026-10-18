from collections.abc import Mapping

import torch
from torch import nn

from mufu.contract import D_MODEL, MODALITIES, _check_keys, _check_tensor


class ModalityProjection(nn.Module):
    """
    Per-encoder linear projection into the shared dimension D_MODEL.

    `in_dims` maps each modality in MODALITIES to its encoder's `out_dim`. Each gets its
    own nn.Linear(in_dims[m], D_MODEL), meaning forward takes {m: (B, in_dims[m])} and
    returns {m: (B, D_MODEL)} for late fusion.
    """

    def __init__(self, in_dims: Mapping[str, int]) -> None:
        super().__init__()
        _check_keys("in_dims", set(in_dims), set(MODALITIES))
        self.in_dims = dict(in_dims)
        self.layers = nn.ModuleDict(
            {m: nn.Linear(in_dims[m], D_MODEL) for m in MODALITIES}
        )

    def forward(
        self, embeddings: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        _check_keys("embeddings", set(embeddings), set(MODALITIES))
        size = len(embeddings[MODALITIES[0]])
        for m in MODALITIES:
            _check_tensor(m, embeddings[m], (size, self.in_dims[m]), "float")
        return {m: self.layers[m](embeddings[m]) for m in MODALITIES}
