# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import scipy.io as sio
import torch


def read_data(data_dir, target_dir):
    data = sio.loadmat(data_dir)['indian_pines_corrected']
    target = sio.loadmat(target_dir)['indian_pines_gt']
    data = data.transpose(2, 0, 1)
    data = normalize(data)
    return data, target


def produce_samples(data, target, prop):
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T  # data size: [height*width, channel]
    target = target.ravel()  # target size: [height*width, ]
    data = data[target != 0]  # remove background
    target = target[ target != 0] - 1 # calss numbers bingin with 0

    sample_num = len(target)
    train_num = round(sample_num*prop)
    train_indices = random.sample(range(0, sample_num), train_num)  # indices of training samples
    test_indices = list(set(np.arange(0, sample_num)).difference(train_indices))  # indices of test samples

    train_data = data[train_indices]
    train_target = target[train_indices]
    test_data = data[test_indices]
    test_target = target[test_indices]

    train_data = torch.from_numpy(train_data).float()
    train_target = torch.from_numpy(train_target).long()
    test_data = torch.from_numpy(test_data).float()
    test_target = torch.from_numpy(test_target).float()

    return train_data, train_target, test_data, test_target


def normalize(data):
    # data: channel*height*width
    data = data.astype(np.float)
    for i in range(len(data)):
        data[i, :, :] -= data[i, :, :].min()
        data[i, :, :] /= data[i, :, :].max()
    return data


def get_one_batch(train_data, train_target, batch_size):
    train_data = torch.split(train_data, batch_size, dim=0)
    train_target = torch.split(train_target, batch_size, dim=0)

    for i in range(len(train_data)):
        yield train_data[i], train_target[i]


if __name__ == '__main__':
    data_dir = './data/Indian_pines_corrected.mat'
    target_dir = './data/Indian_pines_gt.mat'
    prop = 0.2
    batch_size = 30
    data, target = read_data(data_dir, target_dir)
    train_data, train_target, test_data, test_target = produce_samples(data, target, prop)
    # print('train_data', train_data.shape)
    # print('train_target', train_target.shape)
    # print('test_data', test_data.shape)
    # print('test_traget', test_target.shape)
    for data in get_one_batch(train_data, train_target, batch_size):
        print(data[0].shape, data[1].shape)

