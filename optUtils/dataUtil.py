#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/4 18:09
# @Author : LYX-夜光

import numpy as np

from optUtils import set_seed


# 分层打乱下标
def stratified_shuffle_index(y, n_splits, random_state=0):
    if random_state:
        set_seed(random_state)
    # 按标签取出样本，再打乱样本下标
    indexList_label = {}
    for label in set(y):
        indexList_label[label] = np.where(y == label)[0]
        np.random.shuffle(indexList_label[label])
    # 将每个标签的样本分为n_splits份，再按顺序排列每份样本，即按照如下排列：
    # 第1个标签的第1份样本,第2个标签的第1份样本,...,第n个标签的第1份样本,第1个标签的第2份样本,...
    # 每个标签的第1份样本排列后称为第1折，第2份样本排列后称为第2折，...，同时打乱每一折的样本
    indexList = []
    for n in range(n_splits, 0, -1):
        indexList_n = []
        for label in indexList_label:
            split_point = int(len(indexList_label[label]) / n + 0.5)
            indexList_n.append(indexList_label[label][:split_point])
            indexList_label[label] = indexList_label[label][split_point:]
        indexList_n = np.hstack(indexList_n)
        np.random.shuffle(indexList_n)
        indexList.append(indexList_n)
    indexList = np.hstack(indexList)
    return indexList

# 分层打乱样本
def stratified_shuffle_samples(X, y, n_splits, random_state=0):
    indexList = stratified_shuffle_index(y, n_splits, random_state)
    X = [x[indexList] for x in X] if type(X) == list else X[indexList]
    y = y[indexList]
    return X, y
