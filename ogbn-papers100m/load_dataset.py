import json
import scipy
import pickle as pkl
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch.nn.functional as F
import gc
import scipy.sparse as sp
import networkx as nx

def prepare_label_emb(args, g, labels, n_classes, train_idx, valid_idx, test_idx, label_teacher_emb=None):
    print(n_classes)
    print(labels.shape[0])
    import os
    if (not os.path.exists(f'./{args.dataset}_label_0.pt')):
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)
    del labels
    gc.collect()
    res=[]
    import os
    for hop in range(args.label_num_hops):
        if os.path.exists(f'./{args.dataset}_label_{hop}.pt'):
            y=torch.load(f'./{args.dataset}_label_{hop}.pt')
        else:
            y = neighbor_average_labels(g, y.to(torch.float), args)
            torch.save(torch.cat([y[train_idx], y[valid_idx], y[test_idx]], dim=0),f'./{args.dataset}_label_{hop}.pt')

        gc.collect()
        if hop>=args.label_start:
            new_res=y
            #res.append(torch.cat([new_res[train_idx], new_res[valid_idx], new_res[test_idx]], dim=0))
            res.append(new_res)
    return res


def neighbor_average_labels(g, feat, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged labels")
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')

    return feat


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.num_hops + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res

def batched_acc(labels,pred):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)

def get_evaluator(dataset):
    dataset = dataset.lower()
    return batched_acc

def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
#    if dataset=='ogbn-mag':
#        return batched_acc
#    else:
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

def load_dataset(name, device, args):
    """
    Load dataset and move graph and features to device
    """
    if name not in ["ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root=args.root)
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, labels = dataset[0]
    #g = None
    n_classes = dataset.num_classes
    labels = labels.squeeze()

    labels = labels.to(torch.long)
    n_classes = max(labels) + 1
    evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    dataset=args.dataset
    if dataset in ['ogbn-papers100M']:
        data = load_dataset(args.dataset, device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
        #g = dgl.add_reverse_edges(g, copy_ndata=True)
        #feat=g.ndata.pop('feat')
        gc.collect()
        label_emb = None
        if args.use_label:
            label_emb = prepare_label_emb(args, g, labels, n_classes, train_nid, val_nid, test_nid)
        feats=[]
        for i in range(args.num_hops+1):
            feats.append(torch.load(f"./papers100m_feat_{i}.pt"))
        in_feats=feats[0].shape[1]
    
        train_nid = train_nid.to(device)
        val_nid = val_nid.to(device)
        test_nid = test_nid.to(device)
        labels = labels.to(device).to(torch.long)
    return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]),int(in_feats), int(n_classes), \
            train_nid, val_nid, test_nid, evaluator, label_emb
