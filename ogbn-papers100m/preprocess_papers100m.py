import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
import os
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser()
parser.add_argument('--num_hops', type=int, default=6)
parser.add_argument('--root', type=str, default='./')
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-papers100M', root=args.root)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
data = dataset[0]
x = data.x
N = data.num_nodes

import numpy as np
path = './adj_gcn.pt'
print('Making the graph undirected.')
data.edge_index, _ = dropout_adj(
       data.edge_index, p=0, num_nodes=data.num_nodes)
data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
print(data)
row, col = data.edge_index
print('Computing adj...')
adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
adj = adj.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
adj = adj.to_scipy(layout='csr')
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
adj=sparse_mx_to_torch_sparse_tensor(adj)
print('Start processing')
saved = torch.cat((x[train_idx], x[valid_idx], x[test_idx]), dim=0)
torch.save(saved, f'./papers100m_feat_0.pt')
for i in tqdm(range(args.num_hops)):
    x = adj @ x
    saved = torch.cat((x[train_idx], x[valid_idx], x[test_idx]), dim=0)
    torch.save(saved, f'./papers100m_feat_{i+1}.pt')
