# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
from torch import nn, optim
from tools import *
from model import *


# prepare data, set super-parameters
TRAIN_PROP = 0.2
BATCH_SIZE = 2051
EPOCH = 1000
LR = 0.01
TEST_INTERVAL = 1

data_dir = './data/Indian_pines_corrected.mat'
target_dir = './data/Indian_pines_gt.mat'
data, target = read_data(data_dir, target_dir)

# if cuda is avaliable, return values' type is 'torch.cuda.FloatTensor'
train_data, train_target, test_data, test_target = produce_samples(data, target, TRAIN_PROP)

# train model　and save
model = BpNet()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()  # loss function: cross entropy
optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer: adam

loss_list = []
accuracy_list = []
best_accuracy = 0
save_dir = './model_save'
for epoch in range(EPOCH):
    print('Epoch: {}'.format(epoch + 1))

    for idx, data in enumerate(get_one_batch(train_data, train_target, BATCH_SIZE)):
        train_data = data[0]
        train_target = data[1]
        output = model(train_data)
        loss = criterion(output, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % TEST_INTERVAL == 0:
            test_output = model(test_data)
            test_output = test_output.cpu()  # copy cuda tensor to host memory then convert to numpy
            test_target = test_target.cpu()
            test_pred = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((test_pred == test_target.data.numpy()).astype(int).sum()) / \
                       float(test_target.size(0))  # compute accuracy
            accuracy_list.append(accuracy)
            print('Batch: {0} | Train Sample: {1} | Label: {2} | Loss: {3:.8f}　| Accuracy: {4:8f}.'. \
                  format(idx + 1, train_data.size(), train_target.size(), loss.item(), accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = [epoch, idx, loss, best_accuracy]
                state_dict = model.state_dict()
        loss_list.append(loss.item())

plot_curves(loss_list, accuracy_list)
model_name = 'bpnet_' + str(TRAIN_PROP) + '_' + str(BATCH_SIZE) + '_' + str(EPOCH) + '.pkl'
model_dir = os.path.join(save_dir, model_name)
torch.save(state_dict, model_dir)
print('Best Results: ')
print('Epoch: {}  Batch: {}  Loss: {}  Accuracy: {}'.format(*best_state))



