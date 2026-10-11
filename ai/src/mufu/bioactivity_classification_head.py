import torch
from torch import nn

from mufu.contract import N_BIO


class BioactivityClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(N_BIO, N_BIO)

    def forward(self, pic50) -> torch.Tensor:
        return self.linear(pic50)
