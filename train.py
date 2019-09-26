# -*- coding: utf-8 -*-

import os
import pdb
import numpy as np
import scipy.io as iso
import torch
from torch import nn, optim
from tools import *
from model import *


# prepare data, set super-parameters
TRAIN_PROP = 0.2
VAL_PROP = 0.2
BATCH_SIZE = 2051
EPOCH = 20000
LR = 0.001
TEST_INTERVAL = 1

data_dir = './data/Indian_pines_corrected.mat'
target_dir = './data/Indian_pines_gt.mat'
data, target = read_data(data_dir, target_dir)

mask_fname = './data/' + 'train_' + str(TRAIN_PROP) + '_val_' + str(VAL_PROP)
if not os.path.exists(mask_fname) or os.listdir(mask_fname) is None:
    train_mask, val_mask, test_mask = get_masks(target, TRAIN_PROP, VAL_PROP, save_dir='./data')
else:
    train_mask = sio.loadmat(os.path.join(mask_fname, 'train_mask.mat'))['train_mask']
    val_mask = sio.loadmat(os.path.join(mask_fname, 'val_mask.mat'))['val_mask']
    test_mask = sio.loadmat(os.path.join(mask_fname, 'test_mask.mat'))['test_mask']

# if cuda is avaliable, return values' type is 'torch.cuda.FloatTensor'
train_data, train_target = get_samples(data, target, train_mask)
val_data, val_target = get_samples(data, target, val_mask)
test_data, test_target = get_samples(data, target, test_mask)

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
            val_output = model(val_data)
            val_output = val_output.cpu()  # copy cuda tensor to host memory then convert to numpy
            val_target = val_target.cpu()
            test_pred = torch.max(val_output, 1)[1].data.numpy()
            accuracy = float((test_pred == val_target.data.numpy()).astype(int).sum()) / \
                       float(val_target.size(0))  # compute accuracy
            accuracy_list.append(accuracy)
            print('Batch: {0} | Train Sample: {1} | Label: {2} | Loss: {3:.8f}　| Accuracy: {4:8f}.'. \
                  format(idx + 1, train_data.size(), train_target.size(), loss.item(), accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = [epoch, idx, loss, best_accuracy]
                state_dict = model.state_dict()
        loss_list.append(loss.item())

plot_curves(loss_list, accuracy_list)
model_name = 'bpnet_' + str(TRAIN_PROP) + '_' + str(VAL_PROP) + '_' + \
             str(BATCH_SIZE) + '_' + str(EPOCH) + '.pkl'
model_dir = os.path.join(save_dir, model_name)
torch.save(state_dict, model_dir)
print('Best Results: ')
print('Epoch: {}  Batch: {}  Loss: {}  Accuracy: {}'.format(*best_state))



