#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import Data, read_sessions, seq_augument, inputs_target_split
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/diginetica/gowalla')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=2, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
opt = parser.parse_args()
print(opt)


def main():
    data_floder = '../datasets/'
    train_seqs = read_sessions(data_floder + opt.dataset + '/' + 'train.txt')
    test_seqs = read_sessions(data_floder + opt.dataset + '/' + 'test.txt')
    if opt.dataset == 'sample':
        train_seqs = train_seqs[:int(len(train_seqs)/10)]
        test_seqs = test_seqs[:int(len(test_seqs)/10)]
    train_seqs = seq_augument(train_seqs)
    test_seqs = seq_augument(test_seqs)
    print('num of training samples = ', len(train_seqs))
    print('num of test samples = ', len(test_seqs))

    with open(data_floder + opt.dataset + '/' + 'num_items.txt', 'r') as g:
        n_node = int(g.readline()) + 1 # plus 1
        opt.n_node = n_node
    
    train_data = inputs_target_split(train_seqs)
    test_data = inputs_target_split(test_seqs)
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
