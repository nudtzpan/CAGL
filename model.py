#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn, transpose
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        self.W_c = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_l = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_s = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_hl = nn.Linear(self.hidden_size, 1, bias=False)
        self.w_hs = nn.Linear(self.hidden_size, 1, bias=False)
        self.full_linear = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, init_hidden, gnn_hidden, mask):
        U_l = init_hidden
        U_s = gnn_hidden
        C = torch.matmul(U_l, self.W_c(U_s).transpose(2, 1)) # bs * seq_len * seq_len

        H_l = torch.tanh(self.W_l(U_l) + torch.matmul(C.transpose(2, 1), self.W_s(U_s))) # bs * seq_len * hidden_size
        alpha_l = self.w_hl(H_l).squeeze(-1) # bs * seq_len * 1
        alpha_l = torch.exp(alpha_l) * mask.view(mask.shape[0], -1).float()# bs * seq_len
        alpha_l = alpha_l / torch.sum(alpha_l, -1, keepdim=True)
        z_l = torch.sum(alpha_l.unsqueeze(-1) * U_l * mask.view(mask.shape[0], -1, 1).float(), 1) # bs * hidden_size
        l_ht = init_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        H_s = torch.tanh(self.W_s(U_s) + torch.matmul(C, self.W_l(U_l))) # bs * seq_len * hidden_size
        alpha_s = self.w_hs(H_s).squeeze(-1) # bs * seq_len * 1
        alpha_s = torch.exp(alpha_s) * mask.view(mask.shape[0], -1).float()# bs * seq_len
        alpha_s = alpha_s / torch.sum(alpha_s, -1, keepdim=True)
        z_s = torch.sum(alpha_s.unsqueeze(-1) * U_s * mask.view(mask.shape[0], -1, 1).float(), 1) # bs * hidden_size
        s_ht = gnn_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        a = self.full_linear(torch.cat([z_l, l_ht, z_s, s_ht], -1))

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        init_hidden = self.embedding(inputs)
        gnn_hidden = self.gnn(A, init_hidden)
        return init_hidden, gnn_hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    init_hidden, gnn_hidden = model(items, A)
    init_get = lambda i: init_hidden[i][alias_inputs[i]]
    init_seq_hidden = torch.stack([init_get(i) for i in torch.arange(len(alias_inputs)).long()])
    gnn_get = lambda i: gnn_hidden[i][alias_inputs[i]]
    gnn_seq_hidden = torch.stack([gnn_get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(init_seq_hidden, gnn_seq_hidden, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
