import torch
from torch import nn

from mufu.contract import FUSED_DIM, N_TOX


class ToxicityHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(FUSED_DIM, N_TOX)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)
