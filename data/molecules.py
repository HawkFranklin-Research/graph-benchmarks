# data/molecules.py
import torch
import pickle
import os
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, _get_dgl_url
from tqdm import tqdm

class MoleculeDataset(DGLDataset):
    def __init__(self, data_dir, name):
        self.name = name
        self.data_dir = data_dir
        self.url = _get_dgl_url(f'graphforming/benchmarking-gnns/{self.name}.zip')
        
        super(MoleculeDataset, self).__init__(name=name)

    def download(self):
        data_path = os.path.join(self.data_dir, f'{self.name}.zip')
        if not os.path.exists(data_path):
            download(self.url, path=data_path)
            extract_archive(data_path, os.path.join(self.data_dir, self.name))

    def process(self):
        self.train = torch.load(os.path.join(self.data_dir, self.name, f'{self.name}_train.pt'))
        self.val = torch.load(os.path.join(self.data_dir, self.name, f'{self.name}_val.pt'))
        self.test = torch.load(os.path.join(self.data_dir, self.name, f'{self.name}_test.pt'))
        
        # Process node and edge features
        self.num_atom_type = 0
        self.num_bond_type = 0
        if self.train.graph_lists:
            self.num_atom_type = self.train.graph_lists[0].ndata['feat'].shape[1]
            self.num_bond_type = self.train.graph_lists[0].edata['feat'].shape[1]

    def __getitem__(self, idx):
        # This is just a placeholder, the actual data is in self.train, self.val, self.test
        return self.train[idx]

    def __len__(self):
        return len(self.train.graph_lists)
        
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

