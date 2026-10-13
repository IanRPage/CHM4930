import torch
from torch import nn

from mufu.contract import FUSED_DIM, N_BIO


class BioactivityClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(FUSED_DIM, N_BIO)

    def forward(self, pic50) -> torch.Tensor:
        return self.linear(pic50)
