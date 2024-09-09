import os
import math

import torch
from torch_geometric.utils import to_undirected

from src.datasets import Neural, Celegans, Netscience, Pblog, UCsocial, Condmat, KonectDataset
from src.utils.positional_encoding import AddRWPE


# root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def get_data(args):
    dataset_name = args.dataset_name
    test_pct = args.test_pct
    walk_length = args.walk_length
    path = os.path.join(ROOT_DIR, 'data')
    print(f'reading data from: {path}')
    if dataset_name in ('Neural', 'Celegans', 'Netscience', 'Pblog', 'UCsocial', 'Condmat'):
        dataset = eval(dataset_name)(path, dataset_name)
    elif dataset_name in ('Astro', 'Collaboration', 'Congress', 'Usair'):
        dataset = KonectDataset(path, dataset_name)
    else:
        raise ValueError('Invalid dataset name', dataset_name)

    # make random splits
    data = train_test_split_edges(dataset.data, test_pct)

    # adds the random walk positional encoding
    transform = AddRWPE(walk_length, attr_name=None, use_eweight=args.rwpe_use_weight)
    data = transform(data)

    return data


def train_test_split_edges(data, test_ratio):
    row, col = data.edge_index
    edge_weight = data.edge_weight
    del data.edge_index
    del data.edge_weight

    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_weight is not None:
        edge_weight = edge_weight[perm]

    r, c = row[:n_t], col[:n_t]
    data.test_edge_index = torch.stack([r, c], dim=0)
    if edge_weight is not None:
        data.test_edge_weight = edge_weight[:n_t]

    r, c = row[n_t:], col[n_t:]
    data.edge_index = torch.stack([r, c], dim=0)
    if edge_weight is not None:
        out = to_undirected(data.edge_index, edge_weight[n_t:])
        data.edge_index, data.edge_weight = out
    else:
        data.edge_index = to_undirected(data.edge_index)

    return data
