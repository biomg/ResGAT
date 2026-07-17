import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn
from arg_parser import parse_args
import torch_geometric.nn as geom_nn


class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,heads=8):
        super(GraphNet, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out

        # self.conv1 = SAGEConv(num_node_features, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.conv1 = GATConv(num_node_features, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):

        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)

        x1 = F.dropout(x1, p=self.droup_out, training=self.training)
        x2 = self.conv2(x1, edge_index)

        x = x1 + x2   # residual
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)
