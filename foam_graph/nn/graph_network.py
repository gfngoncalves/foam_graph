import torch
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter_add
from torch_geometric.nn import LayerNorm, MetaLayer


class _MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, hidden_dim, with_norm=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.with_norm = with_norm

        if self.with_norm:
            self.norm = LayerNorm(out_dim)
        self.act = ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        for i in range(num_hidden):
            layer = Linear(hidden_dim if i > 0 else in_dim, hidden_dim)
            self.layers.append(layer)

        self.out_layer = Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.act(out)
        out = self.out_layer(out)
        if self.with_norm:
            out = self.norm(out)
        return out


class _NodeModel(torch.nn.Module):
    def __init__(
        self, num_node_features, num_edge_features, out_dim, num_hidden, hidden_dim
    ):
        super(_NodeModel, self).__init__()
        num_node_model_features = num_node_features + num_edge_features

        self.node_mlp = _MLP(
            num_node_model_features, out_dim, num_hidden, hidden_dim, with_norm=True,
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp(out)


class _EdgeModel(torch.nn.Module):
    def __init__(
        self, num_node_features, num_edge_features, out_dim, num_hidden, hidden_dim
    ):
        super(_EdgeModel, self).__init__()
        num_edge_model_features = num_node_features + 2 * num_edge_features

        self.edge_mlp = _MLP(
            num_edge_model_features, out_dim, num_hidden, hidden_dim, with_norm=True,
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out


class GraphNetwork(torch.nn.Module):
    """Graph Network, as proposed in `"Relational inductive biases, 
    deep learning, and graph networks" <https://arxiv.org/abs/1806.01261>`_

    Args:
        num_node_features (int): Dimension of the node feature vector.
        num_edge_features (int): Dimension of the edge feature vector.
        num_targets (int): Dimension of the target vector.
        hidden_channels (int): Dimension of the latent space.
        num_hidden (int): Number of hidden layers in MLPs.
        num_blocks (int): Number of message passing blocks.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_targets: int,
        hidden_channels: int,
        num_hidden: int,
        num_blocks: int,
    ):
        super(GraphNetwork, self).__init__()

        self.node_encoder = _MLP(
            num_node_features,
            hidden_channels,
            num_hidden,
            hidden_channels,
            with_norm=True,
        )
        self.edge_encoder = _MLP(
            num_edge_features,
            hidden_channels,
            num_hidden,
            hidden_channels,
            with_norm=True,
        )

        self.layers = torch.nn.ModuleList()
        for i in range(num_blocks):
            layer = MetaLayer(
                _EdgeModel(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    num_hidden,
                    hidden_channels,
                ),
                _NodeModel(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    num_hidden,
                    hidden_channels,
                ),
            )
            self.layers.append(layer)

        self.decoder = _MLP(hidden_channels, num_targets, num_hidden, hidden_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x_out, edge_attr_out, u = layer(x, edge_index, edge_attr)
            x += x_out
            edge_attr += edge_attr_out

        return self.decoder(x)
