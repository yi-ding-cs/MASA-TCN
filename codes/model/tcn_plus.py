import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalBlockPro(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockPro, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=(0, padding), dilation=(1, dilation)))
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=(0, padding), dilation=(1, dilation)))
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        net = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(net + res)


class TemporalConvNetPro(nn.Module):
    def __init__(self, num_channels, num_eeg_chan=32, freq=6, kernel_size=2, dropout=0.2, early_fusion=True):
        super(TemporalConvNetPro, self).__init__()
        self.early_fusion = early_fusion
        if early_fusion:
            self.fusion_layer = weight_norm(nn.Conv2d(
                in_channels=num_channels[0], out_channels=num_channels[0],
                kernel_size=(num_eeg_chan, 1), stride=(1, 1)
            ))
        else:
            self.fusion_layer = nn.Identity()
        self.space_aware_temporal_layer = nn.Sequential(
            weight_norm(nn.Conv2d(
                in_channels=1, out_channels=num_channels[0],
                kernel_size=(freq, kernel_size), stride=(freq, 1),
                dilation=(1, 2), padding=(0, ((kernel_size - 1) * 2)))),
            Chomp2d((kernel_size - 1) * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            self.fusion_layer
        )
        layers = []
        num_levels = len(num_channels) - 1
        for i in range(num_levels):
            dilation_size = 2 ** (i+2)
            in_channels = num_channels[i] if i == 0 else num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlockPro(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size - 1) * dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        self.space_aware_temporal_layer[0].weight.data.normal_(0, 0.01)
        if self.early_fusion:
            self.fusion_layer.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.space_aware_temporal_layer(x)
        return self.network(x)


class SpaceAwareTemporalBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, num_eeg_chan=32, freq=6, kernel_size=2, dropout=0.2, early_fusion=True):
        super(SpaceAwareTemporalBlock, self).__init__()
        self.early_fusion = early_fusion
        if early_fusion:
            self.fusion_layer = weight_norm(nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=(num_eeg_chan, 1), stride=(1, 1)
            ))
        else:
            self.fusion_layer = nn.Identity()
        self.space_aware_temporal_layer = nn.Sequential(
            weight_norm(nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(freq, kernel_size), stride=(freq, 1),
                dilation=(1, 2), padding=(0, ((kernel_size - 1) * 2)))),
            Chomp2d((kernel_size - 1) * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            self.fusion_layer
        )
        self.init_weights()

    def forward(self, x):
        return self.space_aware_temporal_layer(x)

    def init_weights(self):
        self.space_aware_temporal_layer[0].weight.data.normal_(0, 0.01)
        if self.early_fusion:
            self.fusion_layer.weight.data.normal_(0, 0.01)


class TemporalConvNetProM(nn.Module):
    def __init__(self, num_channels, num_eeg_chan=32, freq=6, kernel_size=[2, 4, 6], dropout=0.2, early_fusion=True):
        super(TemporalConvNetProM, self).__init__()
        self.early_fusion = early_fusion
        self.sa_tcn_1 = SpaceAwareTemporalBlock(
            out_channels=num_channels[0], num_eeg_chan=num_eeg_chan,
            freq=freq, kernel_size=kernel_size[0], dropout=dropout, early_fusion=early_fusion)

        self.sa_tcn_2 = SpaceAwareTemporalBlock(
            out_channels=num_channels[0], num_eeg_chan=num_eeg_chan,
            freq=freq, kernel_size=kernel_size[1], dropout=dropout, early_fusion=early_fusion)

        self.sa_tcn_3 = SpaceAwareTemporalBlock(
            out_channels=num_channels[0], num_eeg_chan=num_eeg_chan,
            freq=freq, kernel_size=kernel_size[2], dropout=dropout, early_fusion=early_fusion)

        layers = []
        num_levels = len(num_channels) - 1
        for i in range(num_levels):
            dilation_size = 2 ** (i+2)
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlockPro(in_channels, out_channels, kernel_size[1], stride=1, dilation=dilation_size,
                                     padding=int((kernel_size[1] - 1) * dilation_size), dropout=dropout)]

        self.OneByOneConv = weight_norm(nn.Conv2d(
                in_channels=3*num_channels[0], out_channels=num_channels[0],
                kernel_size=(1, 1), stride=(1, 1)
            ))
        self.OneByOneConv.weight.data.normal_(0, 0.01)
        self.pure_temporal_layers = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.sa_tcn_1(x)
        x2 = self.sa_tcn_2(x)
        x3 = self.sa_tcn_3(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.OneByOneConv(x)
        return self.pure_temporal_layers(x)
