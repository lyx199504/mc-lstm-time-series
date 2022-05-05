#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/3 22:01
# @Author : LYX-夜光
import time

import numpy as np

import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from skopt import BayesSearchCV

from optUtils import make_dirs, yaml_config
from optUtils.logUtil import logging_config
from optUtils.modelUtil import model_selection


# 机器学习常规训练
def ml_train(X, y, X_test, y_test, model_name, model_param={}, metrics_list=(), model=None):
    """
    :param X: 训练集的特征
    :param y: 训练集的标签
    :param X_test: 测试集的特征
    :param y_test: 测试集的标签
    :param model_name: 模型名称
    :param model_param: 模型参数，可缺省
    :param metrics_list: 多个评价指标，可缺省，默认使用模型自带的评价指标
    :return:
    """

    log_dir = yaml_config['dir']['log_dir']
    cus_param = yaml_config['cus_param']

    if model is None:
        model = model_selection(model_name, **model_param)

    start_time = time.time()

    model.fit(X, y)

    # 获取评价指标
    def get_score(mdl, X, y):
        score_list = []
        if metrics_list:
            y_pred = mdl.predict(X)
            for metrics in metrics_list:
                score_list.append(metrics(y, y_pred))
            return score_list
        score_list.append(mdl.score(X, y))
        return score_list

    train_score_list = get_score(model, X, y)
    train_score_dict = {metrics.__name__: train_score for metrics, train_score in zip(metrics_list, train_score_list)}

    test_score_list = get_score(model, X_test, y_test)
    test_score_dict = {metrics.__name__: val_score for metrics, val_score in zip(metrics_list, test_score_list)}

    run_time = int(time.time() - start_time)

    print("model: %s - train score: %.6f - test score: %.6f - time: %ds" % (
        model_name, train_score_list[0], test_score_list[0], run_time))

    # 配置日志文件
    make_dirs(log_dir)
    logger = logging_config(model_name, log_dir + '/%s.log' % model_name)
    log_message = {
        "cus_param": cus_param,
        "best_param_": model_param,
        "best_score_": test_score_list[0],
        "train_score": train_score_list[0],
        "train_score_dict": train_score_dict,
        "test_score_dict": test_score_dict,
    }
    logger.info(log_message)


# 交叉验证
def cv_train(X, y, model_name, model_param={}, metrics_list=(), model=None):
    """
    :param X: 训练集的特征
    :param y: 训练集的标签
    :param model_name: 模型名称
    :param model_param: 模型参数，可缺省
    :param metrics_list: 多个评价指标，可缺省，默认使用模型自带的评价指标
    :param model: 机器学习或深度学习模型，可缺省，默认根据模型名称获取模型
    :return:
    """

    log_dir = yaml_config['dir']['log_dir']
    cus_param, cv_param = yaml_config['cus_param'], yaml_config['cv_param']

    if model is None:
        model = model_selection(model_name, **model_param)
    if metrics_list:
        model.metrics = metrics_list[0]

    # 计算每一折的评价指标
    def cv_score(mdl, X, y):
        score_list = []
        if metrics_list:
            y_pred = mdl.predict(X)
            for metrics in metrics_list:
                score_list.append(metrics(y, y_pred))
            return score_list
        score_list.append(mdl.score(X, y))
        return score_list

    # 获取每一折的训练和验证分数
    def get_score(mdl, train_index, val_index):
        start_time = time.time()
        mdl.fit(X[train_index], y[train_index])
        train_score_list = cv_score(mdl, X[train_index], y[train_index])
        val_score_list = cv_score(mdl, X[val_index], y[val_index])
        run_time = int(time.time() - start_time)
        print("train score: %.6f - val score: %.6f - time: %ds" % (train_score_list[0], val_score_list[0], run_time))
        return train_score_list, val_score_list

    print("参数设置：%s" % model_param)
    parallel = Parallel(n_jobs=cv_param['workers'], verbose=4)
    k_fold = KFold(n_splits=cv_param['fold'])
    score_lists = parallel(
        delayed(get_score)(model, train, val) for train, val in k_fold.split(X, y))

    train_score_lists = list(map(lambda x: x[0], score_lists))
    val_score_lists = list(map(lambda x: x[1], score_lists))

    train_score_list = np.mean(train_score_lists, axis=0)
    val_score_list = np.mean(val_score_lists, axis=0)

    train_score_dict = {metrics.__name__: train_score for metrics, train_score in zip(metrics_list, train_score_list)}
    val_score_dict = {metrics.__name__: val_score for metrics, val_score in zip(metrics_list, val_score_list)}

    # 配置日志文件
    make_dirs(log_dir)
    logger = logging_config(model_name, log_dir + '/%s.log' % model_name)
    log_message = {
        "cus_param": cus_param,
        "cv_param": cv_param,
        "best_param_": model_param,
        "best_score_": val_score_list[0],
        "train_score": train_score_list[0],
        "train_score_dict": train_score_dict,
        "val_score_dict": val_score_dict,
    }
    logger.info(log_message)


# 贝叶斯搜索
def bayes_search_train(X, y, model_name, model_param, model=None, X_test=None, y_test=None):
    """
    :param X: 训练集的特征
    :param y: 训练集的标签
    :param model_name: 模型名称
    :param model_param: 模型参数
    :param model: 机器学习或深度学习模型，可缺省，默认根据模型名称获取模型
    :param X_test: 测试集的特征，可缺省
    :param y_test: 测试集的标签，可缺省
    :return: 无，输出模型文件和结果日志
    """

    model_dir, log_dir = yaml_config['dir']['model_dir'], yaml_config['dir']['log_dir']
    cus_param, bys_param = yaml_config['cus_param'], yaml_config['bys_param']

    if not model:
        model = model_selection(model_name)

    # 将训练集分为cv折，进行cv次训练得到交叉验证分数均值，最后再训练整个训练集
    bys = BayesSearchCV(
        model,
        model_param,
        n_iter=bys_param['n_iter'],
        cv=bys_param['fold'],
        verbose=4,
        n_jobs=bys_param['workers'],
        random_state=cus_param['seed'],
    )

    bys.fit(X, y)

    make_dirs(model_dir)
    model_path = model_dir + '/%s-%s.model' % (model_name, int(time.time()))
    if 'device' in bys.best_estimator_.get_params():
        bys.best_estimator_.cpu()
        bys.best_estimator_.device = 'cpu'
    model = bys.best_estimator_
    joblib.dump(model, model_path)

    # 配置日志文件
    make_dirs(log_dir)
    logger = logging_config(model_name, log_dir + '/%s.log' % model_name)
    log_message = {
        "cus_param": cus_param,
        "bys_param": bys_param,
        "best_param_": dict(bys.best_params_),
        "best_score_": bys.best_score_,
        "train_score": bys.score(X, y),
        "model_path": model_path,
    }

    # 如果有测试集，则计算测试集分数
    if X_test and y_test:
        log_message.update({"test_score": bys.score(X_test, y_test)})
    logger.info(log_message)

    return model
