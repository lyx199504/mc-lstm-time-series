#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/4 16:25
# @Author : LYX-夜光

import torch
from torch import nn

from dl_models.c_lstm import C_LSTM


class CMC_LSTM(C_LSTM):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "cmc_lstm"

    def create_model(self):
        conv_type, conv_num = 4, 32
        self.cnn_list = nn.ModuleList([])
        for i in range(conv_type):
            kernel_size = i*2 + 1
            padding = i
            pooling_size = 3
            self.cnn_list.append(
                nn.Sequential(
                    nn.Conv1d(1, conv_num, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.Tanh(),
                    nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size),
                )
            )
        self.lstm = nn.LSTM(input_size=conv_type*conv_num, hidden_size=conv_type*conv_num)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=conv_type*conv_num, out_features=conv_num),
            nn.Tanh(),
            nn.Linear(in_features=conv_num, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        Z = []
        for cnn in self.cnn_list:
            Z.append(cnn(X).permute(2, 0, 1))
        Z = torch.cat(Z, dim=-1)
        _, (h, _) = self.lstm(Z)
        y = self.dnn(h.squeeze(0))
        return y
