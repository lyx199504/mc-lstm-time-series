#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/4 19:08
# @Author : LYX-夜光
import sys
import time

import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm

from optUtils import set_seed, make_dirs, yaml_config
from optUtils.logUtil import logging_config

# pytorch随机种子
def pytorch_set_seed(seed):
    if seed:
        set_seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.cuda.manual_seed_all(seed)  # 并行gpu
        torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致

# pytorch深度学习模型
class PytorchModel(nn.Module, BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__()
        self.model_name = "base_dl"
        self.param_search = True  # 默认开启搜索参数功能
        self.save_model = False  # 常规训练中，默认关闭保存模型功能
        self.only_save_last_epoch = False  # 常规训练只保存最后一个epoch

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

        # 优化器、评价指标
        self.optim = torch.optim.Adam
        self.metrics = None
        self.metrics_list = []  # 多个评价指标

    # numpy or list => tensor
    def to_tensor(self, data):
        if type(data) == torch.Tensor or type(data) == list and type(data[0]) == torch.Tensor:
            return data
        if type(data) == list:
            tensor_data = []
            for sub_data in data:
                dataType = torch.float32 if 'float' in str(sub_data.dtype) else torch.int64
                tensor_data.append(torch.tensor(sub_data, dtype=dataType))
        else:
            dataType = torch.float32 if 'float' in str(data.dtype) else torch.int64
            tensor_data = torch.tensor(data, dtype=dataType)
        return tensor_data

    # 训练
    def fit(self, X, y, X_val=None, y_val=None):
        # 设置随机种子
        pytorch_set_seed(self.random_state)
        # 构建模型
        self.create_model()
        # 初始化优化器
        self.optimizer = self.optim(params=self.parameters(), lr=self.learning_rate)
        # 初始化训练集
        X, y = self.to_tensor(X), self.to_tensor(y)
        # 若不进行超参数搜索，则初始化验证集
        if not self.param_search and X_val is not None and y_val is not None:
            X_val, y_val = self.to_tensor(X_val), self.to_tensor(y_val)

        # 训练每个epoch
        pbar = tqdm(range(self.epochs), file=sys.stdout, desc=self.model_name)
        for epoch in pbar:
            self.to(self.device)
            train_loss, train_score, train_score_list = self.fit_epoch(X, y, train=True)
            train_score_dict = {self.metrics.__name__: train_score}
            for i, metrics in enumerate(self.metrics_list):
                train_score_dict.update({metrics.__name__: train_score_list[i]})
            massage_dict = {"train_loss": "%.6f" % train_loss, "train_score": "%.6f" % train_score}

            # 有输入验证集，则计算val_loss和val_score等
            val_loss, val_score, val_score_dict = 0, 0, {}
            if not self.param_search and X_val is not None and y_val is not None:
                val_loss, val_score, val_score_list = self.fit_epoch(X_val, y_val, train=False)
                val_score_dict = {self.metrics.__name__: val_score}
                for i, metrics in enumerate(self.metrics_list):
                    val_score_dict.update({metrics.__name__: val_score_list[i]})
                massage_dict.update({"val_loss": "%.6f" % val_loss, "val_score": "%.6f" % val_score})

            pbar.set_postfix(massage_dict)

            # 不进行超参数搜索，则存储每个epoch的模型和日志
            if not self.param_search:
                if not self.only_save_last_epoch or self.only_save_last_epoch and epoch + 1 == self.epochs:
                    # 存储模型
                    model_path = None
                    if self.save_model:
                        model_dir = yaml_config['dir']['model_dir']
                        make_dirs(model_dir)
                        model_path = model_dir + '/%s-%03d-%s.model' % (self.model_name, epoch + 1, int(time.time()))
                        # 存储模型时，model及其属性device必须保持相同cpu
                        device = self.device
                        self.device = 'cpu'
                        self.to(self.device)
                        joblib.dump(self, model_path)
                        self.device = device
                        self.to(self.device)
                    # 存储日志
                    log_dir = yaml_config['dir']['log_dir']
                    make_dirs(log_dir)
                    logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
                    logger.info({
                        "epoch": epoch + 1,
                        "best_param_": self.get_params(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_score": train_score,
                        "best_score_": val_score,
                        "train_score_dict": train_score_dict,
                        "val_score_dict": val_score_dict,
                        "model_path": model_path,
                    })

    # 每轮拟合
    def fit_epoch(self, X, y, train):
        mean_loss, y_hat = self.fit_step(X, y, train)

        y_numpy = y.cpu().detach().numpy()
        y_hat_numpy = np.hstack(y_hat) if len(y_hat[0].shape) == 1 else np.vstack(y_hat)
        score = self.score(X, y_numpy, y_hat_numpy)
        score_list = self.score_list(X, y_numpy, y_hat_numpy)

        return mean_loss, score, score_list

    # 拟合步骤
    def fit_step(self, X, y=None, train=True):
        self.train() if train else self.eval()

        total_loss, y_hat = 0, []
        indexList = range(0, len(X[0]) if type(X) == list else len(X), self.batch_size)
        for i in indexList:
            if type(X) == list:
                X_batch = [x[i: i + self.batch_size].to(self.device) for x in X]
            else:
                X_batch = X[i: i + self.batch_size].to(self.device)
            y_batch = None if y is None else y[i:i + self.batch_size].to(self.device)
            output = self.forward(X_batch)

            loss = self.loss_fn(output, y_batch, X_batch)
            total_loss += loss.item()

            y_hat_batch = output[0] if type(output) == tuple else output
            y_hat.append(y_hat_batch.cpu().detach().numpy())

            if train:
                loss.backward()  # 梯度计算
                self.optimizer.step()  # 优化更新权值
                self.optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

        mean_loss = total_loss / len(indexList)

        return mean_loss, y_hat

    # 预测结果
    def predict_output(self, X, batch_size, output_all_value=False):
        self.eval()  # 求值模式
        self.to(self.device)
        X = self.to_tensor(X)
        y_hat = []
        for i in range(0, len(X[0]) if type(X) == list else len(X), batch_size):
            if type(X) == list:
                X_batch = [x[i: i + batch_size].to(self.device) for x in X]
            else:
                X_batch = X[i:i + batch_size].to(self.device)
            output = self.forward(X_batch)
            if output_all_value:
                output_list = output if type(output) == tuple else [output]
                y_hat.append([out.cpu().detach().numpy() for out in output_list])
            else:
                y_hat_batch = output[0] if type(output) == tuple else output
                y_hat.append(y_hat_batch.cpu().detach().numpy())
        if output_all_value:
            y_hat_list = []
            for i in range(len(y_hat[0])):
                output_list = [out[i] for out in y_hat]
                output_stack = np.hstack(output_list) if len(y_hat[0][i].shape) == 1 else np.vstack(output_list)
                y_hat_list.append(output_stack)
            y_hat = y_hat_list
        else:
            y_hat = np.hstack(y_hat) if len(y_hat[0].shape) == 1 else np.vstack(y_hat)
        return y_hat

# 深度学习分类器
class DeepLearningClassifier(PytorchModel):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "dl_clf"
        self._estimator_type = "classifier"
        self.label_num = 0

        self.metrics = accuracy_score

    # 组网
    def create_model(self):
        self.fc1 = nn.Linear(in_features=4, out_features=3)
        self.fc2 = nn.Linear(in_features=3, out_features=self.label_num)
        self.relu = nn.ReLU()

    # 损失函数
    def loss_fn(self, output, y_true, X_true):
        y_hat = output[0] if type(output) == tuple else output
        return F.cross_entropy(y_hat, y_true)

    # 前向推理
    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        return y

    def fit(self, X, y, X_val=None, y_val=None):
        self.label_num = len(set(y)) if self.label_num == 0 else self.label_num
        super().fit(X, y, X_val, y_val)

    # 预测分类概率
    def predict_proba(self, X, batch_size=10000):
        y_prob = super().predict_output(X, batch_size)
        return y_prob

    # 预测分类标签
    def predict(self, X, y_prob=None, batch_size=10000):
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        return y_prob.argmax(axis=1)

    # 评价指标
    def score(self, X, y, y_prob=None, batch_size=10000):
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        y_pred = self.predict(X, y_prob, batch_size)
        if self.label_num == 2 and 'auc' in self.metrics.__name__:
            return self.metrics(y, y_prob[:, 1]) if len(y_prob.shape) > 1 else self.metrics(y, y_prob)
        return self.metrics(y, y_pred)

    # 评价指标列表
    def score_list(self, X, y, y_prob=None, batch_size=10000):
        score_list = []
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        y_pred = self.predict(X, y_prob, batch_size)
        for metrics in self.metrics_list:
            if self.label_num == 2 and 'auc' in metrics.__name__:
                score = metrics(y, y_prob[:, 1]) if len(y_prob.shape) > 1 else metrics(y, y_prob)
            else:
                score = metrics(y, y_pred)
            score_list.append(score)
        return score_list

# 深度学习回归器
class DeepLearningRegressor(PytorchModel):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "dl_reg"
        self._estimator_type = "regressor"

        self.metrics = mean_squared_error

    # 组网
    def create_model(self):
        self.fc1 = nn.Linear(in_features=4, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # 损失函数
    def loss_fn(self, output, y_true, X_true):
        y_hat = output[0] if type(output) == tuple else output
        return F.mse_loss(y_hat, y_true)

    # 前向推理
    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.squeeze(-1)
        return y

    # 预测标签
    def predict(self, X, batch_size=10000):
        y_pred = super().predict_output(X, batch_size)
        return y_pred

    # 评价指标
    def score(self, X, y, y_pred=None, batch_size=10000):
        if y_pred is None:
            y_pred = self.predict(X, batch_size)
        return self.metrics(y, y_pred)

    # 评价指标列表
    def score_list(self, X, y, y_pred=None, batch_size=10000):
        score_list = []
        if y_pred is None:
            y_pred = self.predict(X, batch_size)
        for metrics in self.metrics_list:
            score = metrics(y, y_pred)
            score_list.append(score)
        return score_list

# 自编码器
class AutoEncoder(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "ae"
        self.threshold = 0.5  # 正异常阈值
        self.normal = 0  # 正常数据的类别

    # 组网
    def create_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=4)
        )

    # 损失函数
    def loss_fn(self, output, y_true, X_true):
        X_hat = output[0] if type(output) == tuple else output
        return F.mse_loss(X_hat, X_true)

    # 前向推理
    def forward(self, X):
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        return X_hat

    # 每轮拟合
    def fit_epoch(self, X, y, train):
        mean_loss, X_hat = self.fit_step(X, train=train)
        X_numpy, X_hat_numpy = X.cpu().detach().numpy(), np.vstack(X_hat)
        if train:
            # 用训练数据获取阈值范围
            self.threshold = self.get_threshold(X_numpy, X_hat_numpy)

        y_prob = self.get_proba_score(X_numpy, X_hat_numpy)  # y_pred取1的概率
        y_numpy = y.cpu().detach().numpy()

        score = self.score(X, y_numpy, y_prob)
        score_list = self.score_list(X, y_numpy, y_prob)

        return mean_loss, score, score_list

    # 预测得分
    def get_proba_score(self, X, X_hat):
        # 二范数，同np.sqrt(np.sum((X - X_hat) ** 2, axis=1))
        errors = np.linalg.norm(X - X_hat, axis=1, ord=2)
        # 根据误差计算得分，将分数控制在0-1内
        scores = errors / X.shape[1] if self.normal == 0 else 1 / (errors + 1)
        return scores

    # 计算阈值
    def get_threshold(self, X, X_hat):
        return 0.5

    # 预测概率
    def predict_proba(self, X, batch_size=10000):
        X_hat = super().predict_proba(X, batch_size)
        if type(X) == torch.Tensor:
            X = X.cpu().detach().numpy()
        y_prob = self.get_proba_score(X, X_hat)
        return y_prob

    # 预测标签
    def predict(self, X, y_prob=None, batch_size=10000):
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        if self.normal == 0:
            y_pred = np.array([self.normal if score <= self.threshold else 1 - self.normal for score in y_prob])
        else:
            y_pred = np.array([self.normal if score >= self.threshold else 1 - self.normal for score in y_prob])
        return y_pred

# 监督自编码
class SupervisedAutoEncoder(AutoEncoder):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "sae"

    # 每轮拟合
    def fit_epoch(self, X, y, train):
        y_numpy = y.cpu().detach().numpy()
        normal_index = y_numpy == self.normal
        if train:
            self.fit_step(X[normal_index], train=True)  # 只训练正常数据

        mean_loss, X_hat = self.fit_step(X, train=False)  # 不进行训练
        X_numpy, X_hat_numpy = X.cpu().detach().numpy(), np.vstack(X_hat)
        if train:
            # 用正常数据获取正常阈值范围
            self.threshold = self.get_threshold(X_numpy[normal_index], X_hat_numpy[normal_index])

        y_prob = self.get_proba_score(X_numpy, X_hat_numpy)  # y_pred取1的概率

        score = self.score(X, y_numpy, y_prob)
        score_list = self.score_list(X, y_numpy, y_prob)

        return mean_loss, score, score_list

    # 计算阈值
    def get_threshold(self, X, X_hat):
        scores = self.get_proba_score(X, X_hat)
        return max(scores) if self.normal == 0 else min(scores)

# 变分自编码
class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "vae"

    # 组网
    def create_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=4),
            nn.Sigmoid(),
        )

    # 损失函数
    def loss_fn(self, output, y_true, X_true):
        X_hat, mu, log_sigma = output
        BCE = F.binary_cross_entropy(X_hat, X_true, reduction='sum')
        D_KL = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)
        loss = BCE + D_KL
        return loss

    # 前向推理
    def forward(self, X):
        H = self.encoder(X)
        mu, log_sigma = H.chunk(2, dim=-1)
        Z = self.reparameterize(mu, log_sigma)
        X_hat = self.decoder(Z)
        return X_hat, mu, log_sigma

    # 重构Z层：均值+随机采样*标准差
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma * 0.5)
        esp = torch.randn(std.size())
        z = mu + esp * std
        return z

# 监督变分自编码
class SupervisedVariationalAutoEncoder(SupervisedAutoEncoder, VariationalAutoEncoder):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "svae"
