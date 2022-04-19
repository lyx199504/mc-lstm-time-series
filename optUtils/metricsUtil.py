#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/2 12:04
# @Author : LYX-夜光
from sklearn.metrics import f1_score, average_precision_score


def f1_micro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def f1_macro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def f1_weighted_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def pr_auc_score(y_true, y_prob):
    return average_precision_score(y_true, y_prob)
