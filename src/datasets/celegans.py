from typing import Optional, Callable, List

import os.path as osp

import pandas as pd
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url


class Celegans(InMemoryDataset):
    url = 'https://github.com/wave-zuo/SEA/raw/main/data'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['celegans.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        path = osp.join(self.raw_dir, self.raw_file_names[0])
        data = pd.read_csv(path)
        data['2'] = np.exp(-1 / data['2'])
        num_nodes = data[['0', '1']].values.max() + 1
        data = Data(
            edge_index=torch.tensor(data[['0', '1']].values.T, dtype=torch.long),
            edge_weight=torch.tensor(data['2'].values, dtype=torch.float),
            num_nodes=num_nodes,
            node_id=torch.arange(num_nodes, dtype=torch.long)
        )
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
