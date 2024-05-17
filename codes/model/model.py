from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet
from model.tcn_plus import TemporalConvNetPro, TemporalConvNetProM

import math
import os
import torch
from torch import nn

import numpy as np

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_temporal(nn.Module):
    def __init__(self, model_name, num_inputs=192, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1,
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, bidirectional=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        if "tcn" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                            kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                    batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim
            if bidirectional:
                input_dim = hidden_dim * 2

            self.regressor = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        assert len(x.keys()) == 1, "This model is not designed for more than one modalities."

        features = {}
        x = x[list(x.keys())[0]]
        x = x.squeeze(1)
        if "lstm" in self.model_name:
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
            x = self.temporal(x).transpose(1, 2).contiguous()
        batch, time_step, temporal_feature_dim = x.shape
        x = self.regressor(x).contiguous()
        return x


class SA_TCN(nn.Module):
    def __init__(self, model_name, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1, num_eeg_chan=32, freq=6, output_dim=1, early_fusion=True):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.temporal = TemporalConvNetPro(num_channels=cnn1d_channels, num_eeg_chan=num_eeg_chan, freq=freq,
                                           kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate,
                                           early_fusion=early_fusion)
        self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

    def forward(self, x):
        assert len(x.keys()) == 1, "This model is not designed for more than one modalities."
        x = x[list(x.keys())[0]]  # b, 1, t, h
        x = x.transpose(2, 3).contiguous()
        x = self.temporal(x).transpose(1, 3).contiguous()
        x = x.squeeze(-2)
        x = self.regressor(x).contiguous()
        return x


class MSA_TCN(nn.Module):
    def __init__(self, model_name, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1, num_eeg_chan=32, freq=6,
                 output_dim=1, early_fusion=True):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.temporal = TemporalConvNetProM(num_channels=cnn1d_channels, num_eeg_chan=num_eeg_chan, freq=freq,
                                            kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate,
                                            early_fusion=early_fusion)
        self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

    def forward(self, x):
        assert len(x.keys()) == 1, "This model is not designed for more than one modalities."
        x = x[list(x.keys())[0]]  # b, 1, t, h
        x = x.transpose(2, 3).contiguous()
        x = self.temporal(x).transpose(1, 3).contiguous()
        x = x.squeeze(-2)
        x = self.regressor(x).contiguous()
        return x
