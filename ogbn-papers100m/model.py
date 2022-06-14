from layer import *
import numpy as np
class JK_GAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops,label_num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, n_layers_4,act, pre_process=False, residual=False,use_label=False):
        super(JK_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.label_num_hops=label_num_hops
        self.use_label=use_label
        self.residual = residual
        self.prelu = nn.ReLU()
        self.alpha=alpha
        self.res_fc = nn.Linear(nfeat, hidden, bias=False)
        if pre_process:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*hidden, hidden, hidden, n_layers_1, dropout)
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout) for i in range(num_hops)])
        else:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*nfeat, hidden, hidden, n_layers_1, dropout)
            self.lr_att = nn.Linear(nfeat + hidden, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)
        self.pre_process = pre_process
        if use_label:
            self.label_transform=nn.Parameter(torch.FloatTensor(nclass, nclass))
            self.label_jk_ref=FeedForwardNet(
                nclass*label_num_hops, hidden, nclass,n_layers_3, dropout)
            self.label_output = FeedForwardNet(
                nclass, hidden, nclass, n_layers_4, dropout)
            self.label_att=nn.Linear(nclass+nclass,1)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)


    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list=feature_list
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                            input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    W[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        right_1 = self.lr_output(right_1)
        if self.use_label:
            label_list = [self.label_drop(feature) for feature in label_list]
            #print(label_list[0])
            for i in range(1,len(label_list)):
                #label_list=(1-self.alpha)*label_list[i]+self.alpha*F.softmax(torch.mm(label_list[i],self.label_transform),dim=1)
                alpha=np.cos(i*np.pi/(self.label_num_hops*2))
                label_list[i]=(1-alpha)*label_list[i]+alpha*label_list[-1]
            input_list = label_list
            concat_features = torch.cat(input_list, dim=1)
            jk_ref = self.dropout(self.prelu(self.label_jk_ref(concat_features)))
            attention_scores = [self.act(self.label_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                                input_list]
            W = torch.cat(attention_scores, dim=1)
            W = F.softmax(W, 1)
            right_2 = torch.mul(input_list[0], self.att_drop(
                W[:, 0].view(num_node, 1)))
            for i in range(1, self.label_num_hops):
                right_2 = right_2 + \
                    torch.mul(input_list[i], self.att_drop(
                        W[:, i].view(num_node, 1)))
            right_2 = self.label_output(right_2)
            right_1 = right_1+right_2
        return right_1


class R_GAMLP(nn.Module):  # recursive GAMLP
    def __init__(self, nfeat, hidden, nclass, num_hops,label_num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3,n_layers_4, act, pre_process=False, residual=False,use_label=False):
        super(R_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.label_num_hops=label_num_hops
        self.use_label=use_label
        self.residual = residual
        self.prelu = nn.PReLU()
        self.alpha=alpha
        self.res_fc = nn.Linear(nfeat, hidden, bias=False)
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        self.label_drop = nn.Dropout(label_drop)
        if use_label:
            self.label_transform=nn.Parameter(torch.FloatTensor(nclass, nclass))
            self.label_output = FeedForwardNet(
                nclass, hidden, nclass, n_layers_4, dropout)
            self.label_att=nn.Linear(nclass+nclass,1)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        right_1 = self.lr_output(right_1)
        if self.use_label:
            label_list = [self.label_drop(feature) for feature in label_list]
            for i in range(len(label_list)):
#                temp_matrix=torch.mm(label_list[-1],self.label_transform)
                alpha=np.cos(i*np.pi/(self.label_num_hops*2))
                label_list[i]=(1-alpha)*label_list[i]+alpha*label_list[-1]
            input_list = label_list
            attention_scores = []
            attention_scores.append(self.act(self.label_att(
                torch.cat([input_list[0], input_list[0]], dim=1))))
            for i in range(1, self.label_num_hops):
                history_att = torch.cat(attention_scores[:i], dim=1)
                att = F.softmax(history_att, 1)
                history = torch.mul(input_list[0],
                    att[:, 0].view(num_node, 1))
                for j in range(1, i):
                    history = history + \
                    torch.mul(input_list[j],att[:, j].view(num_node, 1))
                attention_scores.append(self.act(self.label_att(
                    torch.cat([history, input_list[i]], dim=1))))
            attention_scores = torch.cat(attention_scores, dim=1)
            attention_scores = F.softmax(attention_scores, 1)
            right_2 = torch.mul(input_list[0],
                attention_scores[:, 0].view(num_node, 1))
            for i in range(1, self.label_num_hops):
                right_2 = right_2 + \
                    torch.mul(input_list[i],
                        attention_scores[:, i].view(num_node, 1))
            #right_2 = label_list[-1]
            right_2 = self.label_output(right_2)
            right_1 = right_1 + right_2
        return right_1


# adapt from https://github.com/facebookresearch/NARS/blob/main/model.py
class WeightedAggregator(nn.Module):
    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(
                torch.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feat_list):  # feat_list k (N,S,D)
        new_feats = []
        for feats, weight in zip(feat_list, self.agg_feats):
            new_feats.append(
                (feats * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats
class NARS_JK_GAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, num_feats, alpha, n_layers_1, n_layers_2, n_layers_3, n_layers_4,act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False,use_label=False):
        super(NARS_JK_GAMLP, self).__init__()
        self.aggregator = WeightedAggregator(num_feats, nfeat, num_hops)
        self.model = JK_GAMLP(nfeat, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                                  label_drop, alpha, n_layers_1, n_layers_2, n_layers_3,n_layers_4, act, pre_process, residual,use_label)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out1 = self.model(feats, label_emb)
        return out1


class NARS_R_GAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, num_feats, alpha, n_layers_1, n_layers_2, n_layers_3,n_layers_4, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False,use_label=False):
        super(NARS_R_GAMLP, self).__init__()
        self.aggregator = WeightedAggregator(num_feats, nfeat, num_hops)
        self.model = R_GAMLP(nfeat, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                                 label_drop, alpha, n_layers_1, n_layers_2, n_layers_3,n_layers_4, act, pre_process, residual,use_label)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out1 = self.model(feats, label_emb)
        return out1
