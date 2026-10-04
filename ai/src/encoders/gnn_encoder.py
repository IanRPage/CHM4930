from collections.abc import Sequence

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool

from featurize.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class GNNEncoder(nn.Module):
    # implementation based on https://forge.coreweave.com/wandb/graph-neural-networks/GIN/reports/What-are-Graph-Isomorphism-Networks---Vmlldzo1MTExMTg5

    def __init__(
        self,
        # selected from BACE1 hyperparameter sweep
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.out_dim = hidden_dims[-1]
        self.pool_norm = nn.BatchNorm1d(self.out_dim)

        dims = (NODE_FEATURE_DIM, *hidden_dims)

        # aggregation layers using GINEConv to incorporate edge features
        # https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.GINEConv.html
        self.mp_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.mp_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(dims[i], dims[i + 1]),
                        nn.ReLU(),
                        nn.Linear(dims[i + 1], dims[i + 1]),
                        nn.ReLU(),
                        nn.BatchNorm1d(dims[i + 1]),
                    ),
                    edge_dim=EDGE_FEATURE_DIM,
                )
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        for conv in self.mp_layers:
            x = conv(x, edge_index, edge_attr=edge_attr)
            # add dropout between layers
            x = F.dropout(x, p=self.dropout, training=self.training)

        # atom embeddings converted to one molecular embedding, retaining atom multiplicity
        x = global_add_pool(x, batch)
        # rescale output so that it doesn't dominate over outputs like from fingerprint
        x = self.pool_norm(x)

        return x
