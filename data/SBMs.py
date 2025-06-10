# data/SBMs.py
import torch
import pickle
import os
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, _get_dgl_url, extract_archive
from tqdm import tqdm

class SBMsDataset(DGLDataset):
    def __init__(self, data_dir, name):
        self.name = name
        self.data_dir = data_dir
        self.url = _get_dgl_url(f'graphforming/benchmarking-gnns/{self.name}.zip')
        
        super(SBMsDataset, self).__init__(name=name)

    def download(self):
        data_path = os.path.join(self.data_dir, f'{self.name}.zip')
        if not os.path.exists(data_path):
            download(self.url, path=data_path)
            extract_archive(data_path, os.path.join(self.data_dir, self.name))

    def process(self):
        data_path = os.path.join(self.data_dir, self.name)
        with open(os.path.join(data_path, f'{self.name}_train.pkl'), 'rb') as f:
            self.train = pickle.load(f)
        with open(os.path.join(data_path, f'{self.name}_val.pkl'), 'rb') as f:
            self.val = pickle.load(f)
        with open(os.path.join(data_path, f'{self.name}_test.pkl'), 'rb') as f:
            self.test = pickle.load(f)

    def __getitem__(self, idx):
        # This is a placeholder
        return self.train[idx]

    def __len__(self):
        return len(self.train)

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.cat(labels).long()
        return batched_graph, labels
