import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from load_dataset import load_dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
import copy
import wandb
from model import R_GAMLP,JK_GAMLP,NARS_JK_GAMLP,NARS_R_GAMLP

def consis_loss(logps, temp, lam):
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim=2)
    avg_p = torch.mean(ps, dim=2)
    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    sharp_p = sharp_p.unsqueeze(2)
    
    mse = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))
    #kl = torch.mean(torch.sum(ps * (torch.log(ps+1e-8) - torch.log(sharp_p+1e-8)), dim=1, keepdim=True))
    mse = lam * mse
    return mse

def consis_loss_mean_teacher(p_t,p_s, temp, lam):
    
    p_t = F.log_softmax(p_t,dim=-1)
    p_t = torch.exp(p_t)
    sharp_p_t = (torch.pow(p_t, 1. / temp) / torch.sum(torch.pow(p_t, 1. / temp), dim=1, keepdim=True)).detach()
    p_s = F.softmax(p_s,dim=1)
    
    #mse = F.mse_loss(sharp_p_t,p_s, reduction = 'mean')
    
    log_sharp_p_t = torch.log(sharp_p_t+1e-8)
    
    loss = torch.mean(torch.sum(torch.pow(p_s - sharp_p_t, 2), dim=1, keepdim=True))
    kl = torch.mean(torch.sum(p_s * (torch.log(p_s+1e-8) - log_sharp_p_t), dim=1, keepdim=True))
    #kldiv = F.kl_div(log_sharp_p_t,p_s,reduction = 'mean')
    
    loss = lam * loss
    return loss,kl

def gen_model_mag(args,num_feats,in_feats,num_classes):
    if args.method=="R_GAMLP":
        return NARS_R_GAMLP(in_feats, args.hidden, num_classes, args.num_hops+1,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual,args.use_label)
    elif args.method=="JK_GAMLP":
        return NARS_JK_GAMLP(in_feats, args.hidden, num_classes, args.num_hops+1,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual,args.use_label)
def gen_model(args,in_size,num_classes):
    if args.method=="R_GAMLP":
        return R_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,args.label_num_hops,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.pre_process,args.residual,args.use_label)
    elif args.method=="JK_GAMLP":
        return JK_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,args.label_num_hops,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.pre_process,args.residual,args.use_label)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def train_rlu_consis(model, train_loader, enhance_loader, optimizer, evaluator, device, xs, labels, label_emb, predict_prob,args,enhance_loader_cons):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []

    total_loss = 0
    iter_num=0
    for idx_1, idx_2 ,idx_3 in zip(train_loader, enhance_loader,enhance_loader_cons):
        logits = []
        idx = torch.cat((idx_1, idx_2,idx_3), dim=0)
        feat_list = [x[idx].to(device) for x in xs]

        batch_feats = [x[idx_3].to(device) for x in xs]
        
        label_list = [x[idx].to(device) for x in label_emb]
        batch_label = [x[idx_3].to(device) for x in label_emb]

        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        
        
        output_att = model(feat_list, label_list)
        
        output_att_f = output_att[len(idx_1)+len(idx_2):]
        
        logits.append(torch.log_softmax(output_att_f, dim=-1))
        logits.append(torch.log_softmax(output_att_f, dim=-1))
        loss_consis = consis_loss(logits, args.tem, args.lam)
        
        output_att = output_att[:len(idx_1)+len(idx_2)]

        L1 = loss_fcn(output_att[:len(idx_1)],  y)*(len(idx_1)*1.0/(len(idx_1)+len(idx_2)))
        teacher_soft = predict_prob[idx_2].to(device)
        teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
        L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)-torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(1).mean()*(len(idx_2)*1.0/(len(idx_1)+len(idx_2)))

        loss = L1 + L3*args.gama+loss_consis
        #loss = L1 + L3*args.gama
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss
        iter_num += 1

    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return loss, approx_acc


def train(model, feats, labels, loss_fcn, optimizer, train_loader,label_emb,evaluator,dataset,use_label):
    model.train()
    device = labels.device
    total_loss = 0
    iter_num=0
    y_true=[]
    y_pred=[]
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        if use_label:
            batch_label = [x[batch].to(device) for x in label_emb]
        else:
            batch_label=[]
        output_att=model(batch_feats,batch_label)
        y_true.append(labels[batch].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1))
        L1 = loss_fcn(output_att, labels[batch].long())
        loss_train = L1
        total_loss += loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num+=1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_pred, dim=0),torch.cat(y_true))
    return loss,acc

def train_rlu(model, train_loader, enhance_loader, optimizer, evaluator, device, xs, labels, label_emb, predict_prob,args):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    iter_num=0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        logits = []
        
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in xs]
        batch_label = [x[idx].to(device) for x in label_emb]
        
        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        output_att= model(feat_list, batch_label)
            
        L1 = loss_fcn(output_att[:len(idx_1)],  y)*(len(idx_1)*1.0/(len(idx_1)+len(idx_2)))
        teacher_soft = predict_prob[idx_2].to(device)
        teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
        L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)-torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(1).mean()*(len(idx_2)*1.0/(len(idx_1)+len(idx_2)))

        loss = L1 + L3*args.gama
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss
        iter_num += 1

    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return loss, approx_acc

def train_scr(model,teacher_model,feats, labels, device, loss_fcn, optimizer, train_loader,enhance_loader,label_emb,evaluator,args,global_step):
    model.train()
    #teacher_model.train()
    total_loss = 0
    total_loss_consis = 0
    total_loss_kl = 0
    total_loss_supervised = 0
    iter_num=0
    y_true=[]
    y_pred=[]
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        batch_label = [x[idx].to(device) for x in label_emb]
        feat_list = [x[idx].to(device) for x in feats]
        feat_list_teacher = [x[idx_2].to(device) for x in feats]
        batch_label_teacher = [x[idx_2].to(device) for x in label_emb]
        
        output_att=model(feat_list,batch_label)
        
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        #L1 = loss_fcn(output_att[:len(idx_1)], labels[idx_1])*(len(idx_1)*1.0/(len(idx_1)+len(idx_2)))
        L1 = loss_fcn(output_att[:len(idx_1)], labels[idx_1].to(device))
        
        with torch.no_grad():
            if args.method == "SAGN":
                mean_t_output,_ = teacher_model(feat_list_teacher,batch_label_teacher)
            else:
                mean_t_output = teacher_model(feat_list_teacher,batch_label_teacher)
        
        student_output = output_att[len(idx_1):]
        
        p_t = mean_t_output
        p_s = student_output
        
        loss_consis,kl_loss = consis_loss_mean_teacher(p_t,p_s,args.tem, args.lam)
        loss_supervised = args.sup_lam*L1
        kl_loss = args.kl_lam*kl_loss
        if args.kl:
            loss_train = loss_supervised+kl_loss
        else:
            loss_train = loss_supervised+loss_consis
        
        total_loss += loss_train
        total_loss_consis += loss_consis
        total_loss_kl += kl_loss
        total_loss_supervised += loss_supervised
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
            
        if args.adap == True:
            alpha = min(1 - 1 / (global_step + 1), args.ema_decay)
        else:
            alpha = args.ema_decay
        for mean_param, param in zip(teacher_model.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)
            
        iter_num+=1
    loss = total_loss / iter_num
    loss_cons = total_loss_consis / iter_num
    loss_kl = total_loss_kl / iter_num
    loss_sup = total_loss_supervised / iter_num
    print(f"loss_cons: {loss_kl}")
    print(f"loss_sup: {loss_sup}")
    acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return loss,acc


@torch.no_grad()
def test(model, feats, labels, test_loader, evaluator, label_emb,dataset,use_label):
    model.eval()
    device = labels.device
    preds = []
    true=[]
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if use_label:
            batch_label = [x[batch].to(device) for x in label_emb]
        else:
            batch_label = []
        true.append(labels[batch].to(torch.long))
        preds.append(torch.argmax(model(batch_feats,batch_label), dim=-1))
    true=torch.cat(true)
    preds = torch.cat(preds, dim=0)
    res = evaluator(preds, true)

    return res

@torch.no_grad()
def gen_output_torch(model, feats, test_loader, device, label_emb):
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        batch_label = [x[batch].to(device) for x in label_emb]
        preds.append(model(batch_feats,batch_label).cpu())
    preds = torch.cat(preds, dim=0)
    return preds
