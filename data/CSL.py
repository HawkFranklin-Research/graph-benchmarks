# data/CSL.py
import torch
import pickle
import os
import dgl
import numpy as np
from dgl.data import DGLDataset
from dgl.data.utils import download, _get_dgl_url, extract_archive
from scipy.sparse import csc_matrix, save_npz, load_npz
from tqdm import tqdm

class CSLDataset(DGLDataset):
    def __init__(self, data_dir, name, split_id=None):
        self.name = name
        self.split_id = split_id
        self.data_dir = data_dir
        self.g_list = []
        self.y_list = []
        self.n_samples = 150
        self.num_node_type = 1
        self.num_edge_type = 1

        super(CSLDataset, self).__init__(name=name,
                                         url=_get_dgl_url('graphforming/benchmarking-gnns/CSL.zip'),
                                         raw_dir=data_dir,
                                         force_reload=False)
    
    def download(self):
        data_path = os.path.join(self.raw_dir, 'CSL.zip')
        if not os.path.exists(data_path):
            download(self.url, path=data_path)
            extract_archive(data_path, os.path.join(self.raw_dir, self.name))

    def process(self):
        # Read the raw graph data
        adj_path = os.path.join(self.raw_dir, self.name, 'graphs.npz')
        y_path = os.path.join(self.raw_dir, self.name, 'y.npy')
        adjs = load_npz(adj_path)
        y = np.load(y_path)
        
        # Reconstruct graphs and labels
        for i in range(self.n_samples):
            g = dgl.from_scipy(csc_matrix(adjs[i]))
            self.g_list.append(g)
            self.y_list.append(y[i])

        # Generate train/val/test splits
        self.train = []
        self.val = []
        self.test = []
        for i in range(5):
            train_idx = np.loadtxt(os.path.join(self.raw_dir, self.name, 'train_split', f'{i}.txt'), dtype=int)
            val_idx = np.loadtxt(os.path.join(self.raw_dir, self.name, 'val_split', f'{i}.txt'), dtype=int)
            test_idx = np.loadtxt(os.path.join(self.raw_dir, self.name, 'test_split', f'{i}.txt'), dtype=int)
            
            self.train.append([(self.g_list[j], self.y_list[j]) for j in train_idx])
            self.val.append([(self.g_list[j], self.y_list[j]) for j in val_idx])
            self.test.append([(self.g_list[j], self.y_list[j]) for j in test_idx])

    def __getitem__(self, i):
        return self.g_list[i], self.y_list[i]

    def __len__(self):
        return self.n_samples

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.tensor(labels).long()
        return batched_graph, labels
