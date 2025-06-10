# layers/gated_gcn_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    GatedGCN: Gated Graph Convolutional Network
    with Equivariant Point-based Encoding (EPE)
    from the paper "Benchmarking Graph Neural Networks"
    https://arxiv.org/pdf/2003.00982.pdf
"""

class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dh_i + Eh_j
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        
        sigma_e = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_e * Bh_j, dim=1) / (torch.sum(sigma_e, dim=1) + 1e-6)
        return {'h': h}

    def forward(self, g, h, e):
        h_in = h
        e_in = e
        
        g.ndata['h'] = h 
        g.edata['e'] = e 
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['Ce'] = self.C(e)
        
        g.update_all(self.message_func, self.reduce_func)
        
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)
            
        h = F.relu(h)
        e = F.relu(e)
        
        if self.residual:
            h = h_in + h
            e = e_in + e
            
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
