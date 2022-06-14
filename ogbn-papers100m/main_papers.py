import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
import uuid
import gc
import wandb

from load_dataset import prepare_data
from utils import *

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def run(args, device):
    checkpt_file = f"./output/{args.dataset}/"+uuid.uuid4().hex
    
    for stage, epochs in enumerate(args.stages):
        if stage > 0:
            predict_prob= torch.load(checkpt_file+'_{}.pt'.format(stage-1))/args.temp
            pred = torch.load(checkpt_file+'_{}.pt'.format(stage-1))
            predict_prob = predict_prob.softmax(dim=1)
            pred = pred.softmax(dim=1)
            train_node_nums=len(train_nid)
            valid_node_nums=len(val_nid)
            test_node_nums=len(test_nid)
            total_num_nodes=train_node_nums+valid_node_nums+test_node_nums
            print("This history model Train ACC is {}".format(evaluator(labels[:train_node_nums],predict_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu())))

            print("This history model Valid ACC is {}".format(evaluator(labels[train_node_nums:train_node_nums+valid_node_nums],predict_prob[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu())))

            print("This history model Test ACC is {}".format(evaluator(labels[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums],predict_prob[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu())))
            confident_nid = torch.arange(len(predict_prob))[
                    predict_prob.max(1)[0] > 0.0]
            extra_confident_nid = confident_nid[confident_nid >= len(
                    train_nid)]            
            print(f'Stage: {stage}, confident nodes: {len(extra_confident_nid)}')
            enhance_idx = extra_confident_nid
            if len(extra_confident_nid) > 0:
                enhance_loader = torch.utils.data.DataLoader(
                        enhance_idx, batch_size=int(args.batch_size*len(enhance_idx)/(len(enhance_idx)+len(train_nid))), shuffle=True, drop_last=False)
                gc.collect()

            
            if args.consis == True:
                confident_nid_cons = torch.arange(len(pred))[
                    pred.max(1)[0] > args.threshold]
                extra_confident_nid_cons = confident_nid_cons[confident_nid_cons >= train_node_nums]
                print(f'Stage: {stage}, confident_cons nodes: {len(extra_confident_nid_cons)}')
                enhance_idx_cons = extra_confident_nid_cons
                if len(extra_confident_nid_cons) > 0:
                    enhance_loader_cons = torch.utils.data.DataLoader(
                        enhance_idx_cons,
                        batch_size=int(args.batch_size * len(enhance_idx_cons) / (len(enhance_idx) + train_node_nums)),
                        shuffle=True, drop_last=False)
            
            
            teacher_probs = torch.zeros(predict_prob.shape[0], predict_prob.shape[1])
            teacher_probs[enhance_idx,:] =   predict_prob[enhance_idx,:]         
        else:
            teacher_probs = None


        with torch.no_grad():
            data = prepare_data(device, args)
        feats, labels, in_size, num_classes, \
                train_nid, val_nid, test_nid, evaluator,label_emb = data
        if stage == 0:
            train_loader = torch.utils.data.DataLoader(
                 torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)), batch_size=int(args.batch_size*len(train_nid)/(len(enhance_idx)+len(train_nid))), shuffle=True, drop_last=False)



        val_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid),len(train_nid)+len(val_nid)), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)+len(val_nid),len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.eval_batch_size,
                shuffle=False, drop_last=False)
        all_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.eval_batch_size,
                shuffle=False, drop_last=False)
        train_node_nums = len(train_nid)
        valid_node_nums = len(val_nid)
        test_node_nums = len(test_nid)
        total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)
        

        model = gen_model(args, in_size, num_classes)
        
        if args.scr == True:
            print("use scr")
            teacher_model = gen_model(args, in_size, num_classes)
            teacher_model = teacher_model.to(device)
            for param in teacher_model.parameters():
                param.detach_()
        
        
        
        model = model.to(device)
        labels=labels.to(device)

        print("# Params:", get_n_params(model))
        loss_fcn = nn.CrossEntropyLoss()
        if args.method == 'JK_GAMLP':
            optimizer_sett = [
            {'params': model.lr_att.parameters(), 'weight_decay': 0, 'lr': 0.0001},
            {'params': model.process.parameters(), 'weight_decay': 0, 'lr': 0.0001},
            {'params': model.lr_jk_ref.parameters(), 'weight_decay': 0, 'lr': 0.0001}, 
            {'params': model.lr_output.parameters(), 'weight_decay': 0, 'lr': 0.0001},
            {'params': model.label_att.parameters(), 'weight_decay': 0, 'lr': 1e-4},
            {'params': model.label_output.parameters(), 'weight_decay': 0, 'lr': 1e-4},
            ]
        else:
            optimizer_sett = [
            {'params': model.lr_att.parameters(), 'weight_decay': 0, 'lr': 0.0001},
            {'params': model.process.parameters(), 'weight_decay': 0, 'lr': 0.0001},
            {'params': model.lr_output.parameters(), 'weight_decay': 0, 'lr': 0.0001},
            {'params': model.label_att.parameters(), 'weight_decay': 0, 'lr': 1e-4},
            {'params': model.label_output.parameters(), 'weight_decay': 0, 'lr': 1e-4},
            ]
        optimizer = torch.optim.Adam(optimizer_sett)
        

        # Start training
        best_epoch = 0
        best_val = 0
        best_test = 0
        count = 0
        if args.scr == True:
            global_step = 0

        for epoch in range(epochs+1):
            gc.collect()
            start = time.time()
            
            if args.scr == False:
                if stage == 0:
                    loss,acc = train(model, feats, labels, loss_fcn, optimizer, train_loader, label_emb,evaluator,args.dataset,args.use_label)
                elif stage != 0 and args.consis == False:
                    loss,acc=train_rlu(model, train_loader, enhance_loader, optimizer, evaluator, device, feats, labels, label_emb, predict_prob,args)
                elif stage != 0 and args.consis == True:
                    loss, acc = train_rlu_consis(model, train_loader, enhance_loader, optimizer, evaluator, device, feats, labels,label_emb, predict_prob, args, enhance_loader_cons)
            else:
                if epoch < (args.warm_up + 1) :
                    loss,acc=train(model, feats, labels, loss_fcn, optimizer, train_loader, label_emb,evaluator,args.dataset,args.use_label)
                else :
                    if epoch == (args.warm_up + 1):
                        print("start scr")
                               
                    if (epoch-1) % args.gap == 0 or epoch == (args.warm_up + 1):
                        preds = gen_output_torch(model, feats, all_loader, device, label_emb)
                        prob_teacher = preds.softmax(dim=1)
                                
                        threshold = args.top - (args.top-args.down) * epoch/epochs
                                
                        confident_nid = torch.arange(len(prob_teacher))[prob_teacher.max(1)[0] > threshold]
                        extra_confident_nid = confident_nid[confident_nid >= train_node_nums]
                        enhance_idx = extra_confident_nid
                        train_loader = torch.utils.data.DataLoader(torch.arange(train_node_nums), batch_size=int(args.batch_size*train_node_nums/(len(enhance_idx)+train_node_nums)), shuffle=True, drop_last=False)
                        enhance_loader = torch.utils.data.DataLoader(
                                enhance_idx, batch_size=int(args.batch_size*len(enhance_idx)/(len(enhance_idx)+train_node_nums)), shuffle=True, drop_last=False)
                            
                    loss, acc = train_scr(model,teacher_model,feats, labels, labels.device, loss_fcn, optimizer, train_loader,enhance_loader, label_emb, evaluator,args, global_step)
                    global_step += 1
            
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f},Train loss: {:.4f}, Train acc: {:.4f} ".format(epoch, end - start,loss,acc*100)
            if epoch % args.eval_every == 0 and epoch >args.train_num_epochs[stage]:
                with torch.no_grad():
                    acc = test(model, feats, labels, val_loader, evaluator,
                                label_emb,args.dataset,args.use_label)
                end = time.time()
                log += "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
                log += "Val {:.4f}, ".format(acc)
                if acc > best_val:
                    best_epoch = epoch
                    best_val = acc
                    best_test = test(model, feats, labels, test_loader, evaluator,
                                    label_emb,args.dataset,args.use_label)
                    torch.save(model.state_dict(),checkpt_file+f'_{stage}.pkl')
                    count = 0
                else:
                    count = count+args.eval_every
                    if count >= args.patience:
                        break
                log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(
                                best_epoch, best_val, best_test)
            print(log)

        print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
                best_epoch, best_val, best_test))
        
        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
        preds = gen_output_torch(model, feats, all_loader, labels.device, label_emb)
        torch.save(preds, checkpt_file+f'_{stage}.pt')


    return best_val, best_test

def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        set_seed(args.seed+i)
        best_val, best_test = run(args, device)
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(f"Average val accuracy: {np.mean(val_accs):.4f}, "
          f"std: {np.std(val_accs):.4f}")
    print(f"Average test accuracy: {np.mean(test_accs):.4f}, "
          f"std: {np.std(test_accs):.4f}")

    return np.mean(test_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCR")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--label-num-hops",type=int,default=3,
                        help="number of hops for label")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed used in the training")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=500000)
    parser.add_argument("--n-layers-1", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-2", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-4", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="number of times to repeat the experiment")
    parser.add_argument("--patience", type=int, default=200,
                        help="early stop of times of the experiment")
    parser.add_argument("--alpha", type=float, default=1,
                        help="temperature of the output prediction")
    parser.add_argument("--beta", type=float, default=0,
                        help="temperature of the output prediction")
    parser.add_argument("--input-drop", type=float, default=0,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.5,
                        help="label feature dropout of model")
    parser.add_argument("--label-start", type=int, default=0,
                        help="label feat dropout of model")
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--use-label", action='store_true', default=False,
                        help="whether to use the label information")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default="relu",
                        help="the activation function of the model")
    parser.add_argument("--method", type=str, default="JK_GAMLP",
                        help="the model to use")
    parser.add_argument("--use-emb", type=str)
    parser.add_argument("--root", type=str, default='')
    parser.add_argument("--train-num-epochs", nargs='+',type=int, default=[100, 100],
                        help="The Train epoch setting for each stage.")
    parser.add_argument("--stages", nargs='+',type=int, default=[300, 300],
                        help="The epoch setting for each stage.")
    parser.add_argument("--temp", type=float, default=1,
                        help="temperature of the output prediction")
    parser.add_argument("--gama", type=float, default=0.5,
                        help="parameter for the KL loss")  
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="the threshold for the node to be added into the model")
    
    #consistency loss
    parser.add_argument("--consis", action='store_true', default=False,
                        help="Whether to use consistency loss")
    parser.add_argument("--con", action='store_true', default=False,
                        help="Whether to use consistency loss")
    parser.add_argument("--tem", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--thres", type=float, default=0.9)
                        
    #scr
    parser.add_argument("--scr", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--adap", action="store_true")
    parser.add_argument("--sup_lam", type=float, default=1.0)
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--kl_lam", type=float, default=0.2)
    parser.add_argument("--top", type=float, default=0.9)
    parser.add_argument("--down", type=float, default=0.7)
    parser.add_argument("--warm_up", type=int, default=60)
    parser.add_argument("--gap", type=int, default=20)
  
    args = parser.parse_args()
    print(args)
    main(args)
