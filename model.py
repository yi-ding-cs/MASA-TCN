from temporal_convolutional_layers import TemporalConvNetPro, TemporalConvNetProM
import torch
from torch import nn

import numpy as np

from torch.nn import Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SA_TCN(nn.Module):
    def __init__(self, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1, num_eeg_chan=32, freq=6, output_dim=1, early_fusion=True, model_type='reg'):
        super().__init__()
        self.output_dim = output_dim
        self.mode = model_type
        if self.mode == 'cls':
            assert output_dim > 1, "This model support at least binary classification. output_dim should > 1."
        self.temporal = TemporalConvNetPro(num_channels=cnn1d_channels, num_eeg_chan=num_eeg_chan, freq=freq,
                                           kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate,
                                           early_fusion=early_fusion)
        self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

    def forward(self, x):
        # x: batch, 1, hidden, seq
        x = self.temporal(x).transpose(1, 3).contiguous()
        x = x.squeeze(-2)
        x = self.regressor(x).contiguous()
        if self.mode == 'cls':
            x = torch.mean(x, dim=1)
        return x


class MASA_TCN(nn.Module):
    def __init__(self, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=[3, 5, 15],
                 cnn1d_dropout_rate=0.1, num_eeg_chan=32, freq=6,
                 output_dim=1, early_fusion=True, model_type='reg'):
        super().__init__()
        self.output_dim = output_dim
        self.mode = model_type
        if self.mode == 'cls':
            assert output_dim > 1, "This model support at least binary classification. output_dim should > 1."
        self.temporal = TemporalConvNetProM(num_channels=cnn1d_channels, num_eeg_chan=num_eeg_chan, freq=freq,
                                            kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate,
                                            early_fusion=early_fusion)
        self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

    def forward(self, x):
        # x: batch, 1, hidden, seq
        x = self.temporal(x).transpose(1, 3).contiguous()
        x = x.squeeze(-2)
        x = self.regressor(x).contiguous()
        if self.mode == 'cls':
            x = torch.mean(x, dim=1)
        return x


if __name__ == "__main__":
    data = torch.randn(1, 1, 192, 96)
    model = MASA_TCN(
        cnn1d_channels=[128, 128, 128],
        cnn1d_kernel_size=[3, 5, 15],
        cnn1d_dropout_rate=0.1,
        num_eeg_chan=32,
        freq=6,
        output_dim=1,
        early_fusion=True,
        model_type='reg')

    output = model(data)
    print("Done")