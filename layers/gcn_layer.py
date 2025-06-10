# layers/gcn_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    GCN: Graph Convolutional Network
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """
    Takes an average over all neighbor node features m and combines it with its own feature h.
    """
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': nodes.data['h'] + accum}

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/1_gcn.html
    """
    def __init__(self, in_feats, out_feats, activation, dropout, batch_norm, residual=False, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin

        if in_feats != out_feats:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, g, feature, snorm_n):
        h_in = feature  # for residual connection
        
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        h = g.ndata['h'] 
        h = h * snorm_n # normalize by square root of src and dst node degree
        h = self.linear(h)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h # residual connection

        h = self.dropout(h)
        return h
