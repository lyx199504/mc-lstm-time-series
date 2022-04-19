#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/11 20:18
# @Author : LYX-夜光
import torch
from torch import nn

from c_lstm import C_LSTM


class MS_CNN(C_LSTM):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "ms_cnn"

    def create_model(self):
        c_lstm_num, c_lstm_hidden_size = 3, 16
        pooling_size = 3
        self.cnn_list = nn.ModuleList([])
        for i in range(c_lstm_num):
            kernel_size = i * 2 + 1
            padding = i
            self.cnn_list.append(
                nn.Sequential(
                    nn.Conv1d(1, c_lstm_hidden_size, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.Tanh(),
                    nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size),
                )
            )
        self.dnn = nn.Sequential(
            nn.Linear(in_features=c_lstm_num*c_lstm_hidden_size*int(self.seq_len/pooling_size), out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        H = []
        for cnn in self.cnn_list:
            X_ = cnn(X)
            H.append(X_.flatten(1))
        H = torch.cat(H, dim=1)
        y = self.dnn(H)
        return y
