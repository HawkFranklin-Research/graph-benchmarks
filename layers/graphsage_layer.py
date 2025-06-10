# layers/graphsage_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    GraphSAGE: 
    Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://arxiv.org/abs/1706.02216
"""

class GraphSageLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_feats != out_feats:
            self.residual = False

        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        if aggregator_type == "maxpool":
            self.fc_pool = nn.Linear(in_feats, in_feats)
        elif aggregator_type == "mean":
            pass # No parameters needed for mean aggregator
        else:
            raise NotImplementedError(f"Aggregator '{aggregator_type}' not supported.")

        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)

    def forward(self, g, h, snorm_n):
        h_in = h # for residual
        h_self = h

        # Aggregate neighbor features
        g.ndata['h'] = h
        if self.aggregator_type == "maxpool":
            g.ndata['h_pool'] = F.relu(self.fc_pool(h))
            g.update_all(fn.copy_src('h_pool', 'm'), fn.max('m', 'neigh'))
        elif self.aggregator_type == "mean":
            g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        
        h_neigh = g.ndata['neigh']
        
        # GraphSAGE-mean convolution
        h = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        
        # Normalization, activation, and dropout
        if self.batch_norm:
            h = self.batchnorm_h(h)
        if self.activation:
            h = self.activation(h)
        if self.residual:
            h = h_in + h
        h = self.dropout(h)
        return h
