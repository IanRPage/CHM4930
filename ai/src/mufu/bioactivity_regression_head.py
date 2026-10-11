import torch
from torch import nn

from mufu.contract import FUSED_DIM, N_BIO


class BioactivityRegressionHead(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(FUSED_DIM, N_BIO),
        )

    def forward(self, x) -> torch.Tensor:
        return self.head(x)
