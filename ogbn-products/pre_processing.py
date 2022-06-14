import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from ogb.nodeproppred import NodePropPredDataset
from cogdl.data import Graph
from cogdl.utils import spmm_cpu


def build_cogdl_graph(name, root):
    dataset = NodePropPredDataset(name=name, root=root)
    graph, y = dataset[0]
    x = torch.tensor(graph["node_feat"]).float().contiguous() if graph["node_feat"] is not None else None
    y = torch.tensor(y.squeeze())
    row, col = graph["edge_index"][0], graph["edge_index"][1]
    row = torch.from_numpy(row)
    col = torch.from_numpy(col)
    edge_index = torch.stack([row, col], dim=0)
    graph = Graph(x=x, edge_index=edge_index, y=y)
    graph.splitted_idx = dataset.get_idx_split()

    return graph


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-products")
parser.add_argument('--num_hops', type=int, default=5)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--giant_path', type=str, default= None)

args = parser.parse_args()
print(args)

graph = build_cogdl_graph(name=args.dataset, root=args.root)
splitted_idx = graph.splitted_idx
train_nid = splitted_idx["train"]
val_nid = splitted_idx["valid"]
test_nid = splitted_idx["test"]

dirs = f"./{args.dataset}/feat/"
if not os.path.exists(dirs):
    os.makedirs(dirs)

if args.giant_path != None:
    graph.x = torch.tensor(np.load(args.giant_path)).float()
    print("Pretrained node feature loaded! Path: {}".format(args.giant_path))

graph.row_norm()
feats = [graph.x]
print("Compute neighbor-averaged feats")
for hop in tqdm(range(1, args.num_hops + 1)):
    feats.append(spmm_cpu(graph, feats[-1]))

for i, x in enumerate(feats):
    feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
    if args.giant_path == None:
        print(f"saved feat_{i}.pt")
        torch.save(feats[i], f'{dirs}/{args.dataset}_feat_{i}.pt')
    else:
        print(f"saved feat_{i}_giant.pt")
        torch.save(feats[i], f'{dirs}/{args.dataset}_feat_{i}_giant.pt')
