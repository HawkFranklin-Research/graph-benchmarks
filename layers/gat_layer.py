# layers/gat_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Petar Veli\u010dkovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li\u00f2, Yoshua Bengio
    (ICLR 2018) https://arxiv.org/abs/1710.10903
"""

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu, dgl_builtin=True):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dgl_builtin = dgl_builtin

        if self.dgl_builtin:
            self.conv = GATConv(in_dim, out_dim // num_heads, num_heads, dropout, dropout, residual_attn=False, activation=activation)
        else:
            raise NotImplementedError("Custom GATLayer not implemented. Use dgl_builtin=True.")
            
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, snorm_n=None):
        h_in = h  # for residual connection
        
        if self.dgl_builtin:
            h = self.conv(g, h).flatten(1)
        else:
            # Implement custom GAT logic here if needed
            pass
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        return h
