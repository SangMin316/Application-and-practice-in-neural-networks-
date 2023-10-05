import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch import autograd
import math
from torch.nn import Parameter


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,stride = 1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out



class FeatureCNN(nn.Module):
    def __init__(self, n_dim=64):
        super(FeatureCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=50, stride= 12, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(32, 64, 2, True)
        self.conv3 = ResBlock(64, 64, 2, True)
        self.conv4 = ResBlock(64, 64, 2, True)
        self.conv5 = torch.nn.Conv2d(64,64,kernel_size=(2,1))
        self.n_dim = n_dim


    def forward(self, x):
        b, c, t = x.shape
        x = x.view(b * c, 1, t)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(b,c,64,-1)
        x = x.view(b,64,c,-1)
        x = self.conv5(x)
        x = x.view(b,-1)
        return x


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.feature_cnn = FeatureCNN()
        self.ReLU = nn.ReLU()
        self.layer1 = nn.Linear(256, 32)
        self.layer2 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.layer1(x)
        out = self.layer2(self.ReLU(out))
        return x, out
        # x <-- latent vector, out <-- model predict