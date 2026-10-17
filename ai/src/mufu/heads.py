import torch
from torch import nn

from mufu.contract import FUSED_DIM, N_BIO, N_TOX


class BioactivityClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(FUSED_DIM, N_BIO)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)


class BioactivityRegressionHead(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(FUSED_DIM, N_BIO),
        )

    def forward(self, x) -> torch.Tensor:
        return self.head(x)


class ToxicityHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(FUSED_DIM, N_TOX)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)
