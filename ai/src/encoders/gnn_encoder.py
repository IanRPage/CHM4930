from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool

from featurize.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class GNNEncoder(nn.Module):
    def __init__(
        self,
        # selected from BACE1 hyperparameter sweep
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.out_dim = hidden_dims[-1]

        # initialization layer using GINEConv to incorproate edge features
        # https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.GINEConv.html
        self.initialization = GINEConv(
            nn.Sequential(
                nn.Linear(NODE_FEATURE_DIM, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[0]),
            ),
            edge_dim=EDGE_FEATURE_DIM,
        )

        ## Aggregation Layers
        self.mp_layers = torch.nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            hidden = hidden_dims[i]
            self.mp_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i - 1], hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
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
        x = self.initialization(x, edge_index, edge_attr=edge_attr)
        # add dropout between layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.mp_layers:
            x = conv(x, edge_index, edge_attr=edge_attr)
            # add dropout between layers
            x = F.dropout(x, p=self.dropout, training=self.training)

        # atom embeddings converted to one molecular embedding, retaining atom multiplicity
        x = global_add_pool(x, batch)
        return x
