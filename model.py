# -*- coding: utf-8 -*-

import numpy as np
from torch import nn


class BpNet(nn.Module):
    def __init__(self):
        super(BpNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(200, 80),
            nn.Sigmoid(),
            nn.Linear(80, 50),
            nn.Sigmoid(),
            nn.Linear(50, 30),
            nn.Sigmoid(),
            nn.Linear(30, 16)
        )

    def forward(self, x):
        y_hat = self.fc(x)
        return y_hat


