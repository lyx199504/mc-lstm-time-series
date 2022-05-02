#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/6 16:02
# @Author : LYX-夜光
import joblib
from torch import nn
import numpy as np

from optUtils import yaml_config
from optUtils.logUtil import get_rank_param, logging_config
from optUtils.pytorchModel import DeepLearningClassifier


class C_LSTM(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60):
        super(C_LSTM, self).__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "c_lstm"
        self.label_num = 2  # 二分类
        self.seq_len = seq_len

    def create_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64)
        self.dnn = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.permute(2, 0, 1)
        _, (h, _) = self.lstm(X)
        y = self.dnn(h.squeeze(0))
        return y

    # # 拟合步骤
    # def fit_step(self, X, y=None, train=True):
    #     y_ = y.clone()
    #     y_[y_ > 0] = 1
    #     return super().fit_step(X, y_, train)

    # 测试数据，将验证集中前k个最高分数的模型用于测试，最终分数为k个测试分数的均值
    def test_score(self, X, y, k=1):
        model_param_list = get_rank_param(
            self.model_name,
            key_list=['best_score_', 'train_score', 'epoch'],
            reverse_list=[True, True, True],
        )[:k]
        log_dir = yaml_config['dir']['log_dir']
        logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
        logger.info({
            "===================== Test scores of the top %d validation models =====================" % k
        })
        test_score_lists = []
        for model_param in model_param_list:
            mdl = joblib.load(model_param['model_path'])
            mdl.device = self.device
            test_score = mdl.score(X, y, batch_size=512)
            test_score_list = mdl.score_list(X, y, batch_size=512)
            test_score_lists.append([test_score] + test_score_list)
            test_score_dict = {self.metrics.__name__: test_score}
            for i, metrics in enumerate(self.metrics_list):
                test_score_dict.update({metrics.__name__: test_score_list[i]})

            logger.info({
                "select_epoch": model_param['epoch'],
                "test_score_dict": test_score_dict,
            })
        logger.info({
            "===================== Mean test score ====================="
        })
        mean_score_list = np.mean(test_score_lists, axis=0)
        mean_score_dict = {self.metrics.__name__: mean_score_list[0]}
        for i, metrics in enumerate(self.metrics_list):
            mean_score_dict.update({metrics.__name__: mean_score_list[i+1]})
        logger.info({
            "mean_score_dict": mean_score_dict,
        })

