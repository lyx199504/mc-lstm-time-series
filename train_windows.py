#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/20 17:41
# @Author : LYX-夜光
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dl_models.c_lstm import C_LSTM
from dataPreprocessing import getDataset, standard
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index
from dl_models.imc_lstm import IMC_LSTM
from dl_models.cmc_lstm import CMC_LSTM
from dl_models.smc_lstm import SMC_LSTM

if __name__ == "__main__":
    seq_len_list = [20, 40, 80, 100]
    dataset_list = ['realAdExchange', 'realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'A1Benchmark', 'realTweets']
    model_list = [C_LSTM, IMC_LSTM, CMC_LSTM, SMC_LSTM]

    for seq_len in seq_len_list:
        for dataset_name in dataset_list:
            for model_clf in model_list:
                X, y, r = getDataset(dataset_name, seq_len=seq_len, pre_list=[standard])

                seed, fold = yaml_config['cus_param']['seed'], yaml_config['cv_param']['fold']
                # 根据r的取值数量分层抽样
                shuffle_index = stratified_shuffle_index(r, n_splits=fold, random_state=seed)
                X, y = X[shuffle_index], y[shuffle_index]

                P, total = sum(y > 0), len(y)
                print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))
                train_point, val_point = int(len(X) * 0.6), int(len(X) * 0.8)

                model = model_clf(learning_rate=0.001, batch_size=512, epochs=500, random_state=1, seq_len=seq_len)
                model.model_name += "_%s_%s" % (seq_len, dataset_name)
                model.param_search = False
                model.save_model = True
                model.device = 'cuda'
                model.metrics = f1_score
                model.metrics_list = [recall_score, precision_score, accuracy_score]
                model.fit(X[:train_point], y[:train_point], X[train_point:val_point], y[train_point:val_point])
                model.test_score(X[val_point:], y[val_point:])
