#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/11 20:03
# @Author : LYX-夜光

from torch import nn

from c_lstm import C_LSTM


class CNN(C_LSTM):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "cnn"

    def create_model(self):
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.dnn = nn.Sequential(
            nn.Linear(in_features=int(self.seq_len/3)*16, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.conv(X)
        X = X.flatten(1)
        y = self.dnn(X)
        return y
