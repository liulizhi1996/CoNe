from typing import Tuple

import torch
from torch import Tensor
from torch.nn import ModuleList, Linear, BatchNorm1d, Embedding
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_sparse import SparseTensor

from src.models.mlp import MLP
from src.models.gps_conv import GPSConv


def elem2spm(element: Tensor, sizes: Tuple[int, int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    elem = torch.bitwise_left_shift(spm.storage.row(), 32).add_(spm.storage.col())
    return elem


class CoNe(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        node_embedding: Embedding = None,
        attn_type: str = 'sga',
        num_heads: int = 1,
        gnn_dropout: float = 0,
        attn_dropout: float = 0,
        mlp_dropout: float = 0
    ):
        super().__init__()

        self.node_embedding = node_embedding

        self.pe_norm = BatchNorm1d(in_channels)
        self.pe_lin = Linear(in_channels, hidden_channels)

        channels = hidden_channels
        if self.node_embedding is not None:
            channels += node_embedding.embedding_dim

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = GPSConv(channels, GCNConv(channels, channels),
                           heads=num_heads, dropout=gnn_dropout,
                           attn_type=attn_type, attn_kwargs={'dropout': attn_dropout})
            self.convs.append(conv)

        self.content_mlp = MLP([channels, channels],
                               dropout=mlp_dropout, norm=False)
        self.structure_mlp = MLP([channels, channels],
                                 dropout=mlp_dropout, norm=False)
        self.out = MLP([2 * channels, channels, 1],
                       dropout=mlp_dropout, norm=True, plain_last=True)

        self.aggr = SoftmaxAggregation(learn=True)

    def content_feature_forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor
    ):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        return x

    def structure_feature_forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor
    ):
        num_nodes = max(z.size(0), edge_index.max() + 1,
                        edge_label_index.max() + 1)
        num_edges = edge_label_index.size(1)

        adj = SparseTensor.from_edge_index(edge_index,
                                           sparse_sizes=(num_nodes, num_nodes))
        source_neighbors = adj[edge_label_index[1]]
        target_neighbors = adj[edge_label_index[0]]

        element1 = spm2elem(source_neighbors)
        element2 = spm2elem(target_neighbors)
        if element2.shape[0] > element1.shape[0]:
            element1, element2 = element2, element1

        idx = torch.searchsorted(element1[:-1], element2)
        mask = (element1[idx] == element2)
        ret_elem = element2[mask]
        common_neighbors = elem2spm(ret_elem, (num_edges, num_nodes))

        idx = common_neighbors.storage.row()
        nbr = common_neighbors.storage.col()
        structure_feature = self.aggr(z[nbr], idx)
        if structure_feature.size(0) < num_edges:
            structure_feature = F.pad(
                structure_feature,
                (0, 0, 0, num_edges - structure_feature.size(0))
            )
        return structure_feature

    def predictor(
        self,
        content: Tensor,
        structure: Tensor,
        edge_label_index: Tensor
    ):
        content = content[edge_label_index[0]] * content[edge_label_index[1]]
        content = self.content_mlp(content)
        structure = self.structure_mlp(structure)
        h = torch.cat((content, structure), dim=-1)
        return self.out(h)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_label_index: Tensor
    ):
        x = self.pe_lin(self.pe_norm(x))
        if self.node_embedding is not None:
            n_emb = self.node_embedding.weight
            x = torch.cat([x, n_emb], dim=1)

        content = self.content_feature_forward(x, edge_index, edge_weight)
        structure = self.structure_feature_forward(
            content, edge_index, edge_label_index)
        return self.predictor(content, structure, edge_label_index)
