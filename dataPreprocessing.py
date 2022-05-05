#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/6 12:28
# @Author : LYX-夜光
from pathlib import Path

import numpy as np
import pandas as pd

from optUtils import read_json

# 标准化
def standard(value):
    mean_value, std_value = value.mean(), value.std()
    return (value - mean_value) / std_value

# 归一化
def norm(value):
    max_value, min_value = value.max(), value.min()
    diff_value = max_value - min_value
    return (value - min_value) / diff_value

def reset_feat(value):
    grow_value = []
    value_median = np.median(value)
    for i in range(0, len(value), 2):
        if i+1 < len(value):
            abs_value1 = abs(value[i] - value_median)
            abs_value2 = abs(value[i+1] - value_median)
            grow_value.append(value[i] if abs_value1 > abs_value2 else value[i+1])
    return grow_value


# 获取NAB数据集，将异常数据定为正例
def getNabDataset(dataset_name, seq_len=50, move_pace=1, pre_list=(standard,), deal_list=()):
    """
    :param dataset_name: 数据集名称
    :param seq_len: 滑动窗口序列长度
    :param move_pace: 滑动窗口步长
    :param pre_list: 编码成滑动窗口之前的数据预处理
    :param deal_list: 编码成滑动窗口之后的数据处理
    :return: X, y
    """
    X, y = [], []
    r = []
    labels = read_json("./datasets/Numenta Anomaly Benchmark/labels/combined_labels.json")
    # windows = read_json("./datasets/NAB/labels/combined_windows.json")
    path_list = list(Path("./datasets/Numenta Anomaly Benchmark/data/%s" % dataset_name).glob("*.csv"))
    for i, path in enumerate(path_list):
        raw_data = pd.read_csv(path, index_col='timestamp')
        raw_value = raw_data.value
        for pre in pre_list:
            raw_value = pre(raw_value)
        raw_data['label'] = 0
        # for window in windows[dataset_name + "/%s" % path.name]:
        #     start, end = window[0].split('.')[0], window[1].split('.')[0]
        #     raw_data.loc[start: end, 'label'] = 1
        for timestamp in labels[dataset_name + "/%s" % path.name]:
            raw_data.loc[timestamp, 'label'] = 1

        # 构造时间序列特征
        s_point, e_point = 0, len(raw_value) - seq_len + 1
        X_ = np.array([raw_value[ix: ix + seq_len] for ix in range(s_point, e_point, move_pace)])
        for deal in deal_list:
            X_ = np.array([deal(value) for value in X_])
        X.append(X_)

        # 构造时间序列标签及其异常位置
        raw_label = raw_data.label
        # y_ = np.array([np.where(raw_label[ix: ix + seq_len] == 1)[0][0] + 1 if sum(raw_label[ix: ix + seq_len]) > 0 else 0 for ix in range(s_point, e_point, move_pace)])
        y_ = np.array([1 if sum(raw_label[ix: ix + seq_len]) > 0 else 0 for ix in range(s_point, e_point, move_pace)])
        y.append(y_)

        r_ = np.array([1 if sum(raw_label[ix: ix + seq_len]) > 0 else 0 for ix in range(s_point, e_point, move_pace)])
        r_ = r_ * 1000 + i
        r.append(r_)

    X, y, r = np.vstack(X).astype('float32'), np.hstack(y).astype('int32'), np.hstack(r).astype('int32')

    print("构造数据集完毕，数据集大小为：%s..." % str(X.shape))
    return X, y, r


# 获取雅虎数据集，将异常数据定为正例
def getWebscopeS5Dataset(dataset_name, seq_len=50, move_pace=1, pre_list=(standard,), deal_list=()):
    X, y = [], []
    r = []
    path_list = list(Path("./datasets/Yahoo! Webscope S5/%s" % dataset_name).glob("*.csv"))
    for i, path in enumerate(path_list):
        raw_data = pd.read_csv(path)
        try:
            raw_value = raw_data.value.values
            for pre in pre_list:
                raw_value = pre(raw_value)
        except:
            continue
        # 构造时间序列特征
        s_point, e_point = 0, len(raw_value) - seq_len + 1
        X_ = np.array([raw_value[ix: ix + seq_len] for ix in range(s_point, e_point, move_pace)])
        for deal in deal_list:
            X_ = np.array([deal(value) for value in X_])
        X.append(X_)

        # 构造时间序列标签
        try:
            raw_label = raw_data.is_anomaly
        except:
            raw_label = raw_data.anomaly
        # y_ = np.array([np.where(raw_label[ix: ix + seq_len] == 1)[0][0] + 1 if sum(raw_label[ix: ix + seq_len]) > 0 else 0 for ix in range(s_point, e_point, move_pace)])
        y_ = np.array([1 if sum(raw_label[ix: ix + seq_len]) > 0 else 0 for ix in range(s_point, e_point, move_pace)])
        y.append(y_)

        r_ = np.array([1 if sum(raw_label[ix: ix + seq_len]) > 0 else 0 for ix in range(s_point, e_point, move_pace)])
        r_ = r_ * 1000 + i
        r.append(r_)

    X, y, r = np.vstack(X).astype('float32'), np.hstack(y).astype('int32'),  np.hstack(r).astype('int32')

    print("构造数据集完毕，数据集大小为：%s..." % str(X.shape))
    return X, y, r

# 根据数据集名称获取数据集
def getDataset(dataset_name, seq_len=50, move_pace=1, pre_list=(standard,), deal_list=()):
    if dataset_name.startswith('A'):
        return getWebscopeS5Dataset(dataset_name, seq_len, move_pace, pre_list, deal_list)
    else:
        return getNabDataset(dataset_name, seq_len, move_pace, pre_list, deal_list)