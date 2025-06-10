# data/data.py
import os
import pickle
import torch
from scipy.spatial.distance import cdist

from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.CYCLES import CYCLES_Dataset
from data.graphtheoryprop import GraphTheoryPropDataset
from data.CSL import CSLDataset
from data.superpixels import SuperPixDataset
from data.TUs import TUsDataset
from data.TSP import TSPDataset
from data.COLLAB import COLLABDataset
from data.WikiCS import WikiCSDataset

from dgl.data import download as dgl_download
from dgl.data.utils import inférieure_url, extract_archive

class LoadData:
    def __init__(self, data_dir, dataset_name):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """
        Loading datasets
        """
        if self.dataset_name == 'AQSOL':
            return MoleculeDataset(self.data_dir, self.dataset_name)
        elif self.dataset_name == 'ZINC':
            return MoleculeDataset(self.data_dir, self.dataset_name)
        elif self.dataset_name == 'SBM_CLUSTER':
            return SBMsDataset(self.data_dir, 'SBM_CLUSTER')
        elif self.dataset_name == 'SBM_PATTERN':
            return SBMsDataset(self.data_dir, 'SBM_PATTERN')
        elif self.dataset_name == 'CYCLES':
            return CYCLES_Dataset(self.data_dir, 'CYCLES')
        elif self.dataset_name == 'GraphTheoryProp':
            return GraphTheoryPropDataset(self.data_dir, 'GraphTheoryProp')
        elif self.dataset_name == 'CSL':
            return CSLDataset(self.data_dir, 'CSL')
        elif self.dataset_name == "MNIST" or self.dataset_name == "CIFAR10":
            return SuperPixDataset(self.data_dir, self.dataset_name)
        elif self.dataset_name in ['DD', 'ENZYMES', 'PROTEINS_full']:
            return TUsDataset(self.data_dir, self.dataset_name)
        elif self.dataset_name == 'TSP':
            return TSPDataset(self.data_dir, self.dataset_name)
        elif self.dataset_name == 'COLLAB':
            return COLLABDataset(self.data_dir, self.dataset_name)
        elif self.dataset_name == 'WikiCS':
            return WikiCSDataset(self.data_dir, self.dataset_name)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _add_positional_encodings(self, pos_enc_dim):
        """
        Loading positional encodings
        """
        # We use Laplacian PE for PNA and GSN.
        self.dataset.train.graph_lists = [self._laplacian_positional_encoding(g, pos_enc_dim) for g in self.dataset.train.graph_lists]
        self.dataset.val.graph_lists = [self._laplacian_positional_encoding(g, pos_enc_dim) for g in self.dataset.val.graph_lists]
        self.dataset.test.graph_lists = [self._laplacian_positional_encoding(g, pos_enc_dim) for g in self.dataset.test.graph_lists]

    def _laplacian_positional_encoding(self, g, pos_enc_dim):
        """
        Graph positional encoding v/ Laplacian eigenvectors
        """
        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = g.number_of_nodes()
        D = torch.diag(torch.sum(torch.from_numpy(A.toarray()), 1))
        L = D - torch.from_numpy(A.toarray())
        
        # Eigenvectors
        eigval, eigvec = torch.linalg.eigh(L)
        
        # Keep smallest non-zero eig
        idx = (eigval > 1e-6).nonzero().squeeze()
        
        g.ndata['pos_enc'] = eigvec[:, idx[:pos_enc_dim]].float()
        
        return g

