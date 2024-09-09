from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.norm.batch_norm import BatchNorm


class MLP(torch.nn.Module):
    def __init__(
        self,
        channel_list: Optional[Union[List[int], Tuple[int]]] = None,
        dropout: Union[float, List[float]] = 0.,
        norm: bool = False,
        plain_last: bool = False,
        bias: Union[bool, List[bool]] = True
    ):
        super().__init__()

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = F.leaky_relu
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list) - 1})")
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list) - 1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm:
                norm_layer = BatchNorm(hidden_channels)
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'
