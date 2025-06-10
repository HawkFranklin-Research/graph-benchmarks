# layers/gin_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    GIN: Graph Isomorphism Network
    How Powerful are Graph Neural Networks? (ICLR 2019)
    https://arxiv.org/abs/1810.00826
"""

class GINLayer(nn.Module):
    def __init__(self, apply_func, aggr_type, dropout, batch_norm, residual=False, init_eps=0, learn_eps=True):
        super().__init__()
        self.apply_func = apply_func
        
        self.aggr_type = aggr_type
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))

        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        in_dim = apply_func.mlp.in_features
        out_dim = apply_func.mlp.out_features
        
        if in_dim != out_dim:
            self.residual = False

        self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        if learn_eps:
            self.eps.requires_grad = True

        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, snorm_n):
        h_in = h
        
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), self._reducer(msg='m', out='neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        
        if self.apply_func is not None:
            h = self.apply_func(h)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        h = F.relu(h)

        if self.residual:
            h = h_in + h
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

class ApplyNodeFunc(nn.Module):
    """
    This is the MLP applied to the aggregated node features in GIN.
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        return self.mlp(h)

