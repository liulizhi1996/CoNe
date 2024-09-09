import torch
import torch.nn as nn


class SGALayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGALayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, x_src, x_dst):
        q, k, v = self.lin_q(x_src), self.lin_k(x_dst), self.lin_v(x_dst)
        q = q / torch.norm(q, p=2)
        k = k / torch.norm(k, p=2)
        N = x_dst.size(0)

        up = torch.einsum('ij,ik->jk', k, v)
        up = torch.einsum('ij,jk->ik', q, up)
        up += N * v

        down = k.sum(dim=0)
        down = torch.einsum('ij,j->i', q, down)
        down += N * torch.ones_like(down)
        down.unsqueeze_(dim=-1)

        attn = up / down
        return attn
