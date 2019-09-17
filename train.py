# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch
from torch import nn, optim
from tools import *
from model import *


# prepare data, set super-parameters
TRAIN_PROP = 0.6
BATCH_SIZE = 300
EPOCH = 1000
LR = 0.01
TEST_INTERVAL = 1

data_dir = './data/Indian_pines_corrected.mat'
target_dir = './data/Indian_pines_gt.mat'
data, target = read_data(data_dir, target_dir)
train_data, train_target, test_data, test_target = produce_samples(data, target, TRAIN_PROP)

# train model
model = BpNet()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()  # loss function: cross entropy
optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer: adam

for epoch in range(EPOCH):
    print('Epoch: {}'.format(epoch + 1))

    for idx, data in enumerate(get_one_batch(train_data, train_target, BATCH_SIZE)):
        train_data = data[0]
        train_target = data[1]
        if torch.cuda.is_available():
            train_data = train_data[0].cuda()
            train_target = train_target[1].cuda()
        output = model(train_data)
        loss = criterion(output, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % TEST_INTERVAL == 0:
            test_output = model(test_data)
            test_pred = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((test_pred == test_target.data.numpy()).astype(int).sum()) / \
                       float(test_target.size(0))  # compute accuracy
            print('Batch: {0} | Train Sample: {1} | Label: {2} | Loss: {3:.8f}　| Accuracy: {4:8f}.'. \
                  format(idx + 1, train_data.size(), train_target.size(), loss.item(), accuracy))




