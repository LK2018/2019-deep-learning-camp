# -*- coding: utf-8 -*-

import sys
import pdb
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
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
    target = target[target != 0] - 1 # class numbers bingin with 0
    data_tmp = []
    target_tmp = []
    for i in range(max(target) + 1):
        data_tmp.append(data[target == i])
        target_tmp.append(target[target == i])

    for i in range(len(data_tmp)):
        sample_num = len(data_tmp[i])
        train_num = int(round(sample_num*prop))
        train_ind = random.sample(range(0, sample_num), train_num)  # indices of training samples
        test_ind = list(set(np.arange(0, sample_num)).difference(train_ind))  # indices of test samples
        train_i = np.hstack((data_tmp[i][train_ind], target_tmp[i][train_ind][:, np.newaxis]))
        test_i = np.hstack((data_tmp[i][test_ind], target_tmp[i][test_ind][:, np.newaxis]))
        if i == 0:
            train = train_i
            test = test_i
        else:
            train = np.vstack((train, train_i))
            test = np.vstack((test, test_i))
    np.random.shuffle(train)
    np.random.shuffle(test)

    train_data = train[:, :-1]
    train_target = train[:, -1]
    test_data = test[:, :-1]
    test_target = test[:, -1]

    train_data = torch.from_numpy(train_data).float()
    train_target = torch.from_numpy(train_target).long()
    test_data = torch.from_numpy(test_data).float()
    test_target = torch.from_numpy(test_target).long()

    if torch.cuda.is_available():
        train_data = train_data.cuda()
        train_target = train_target.cuda()
        test_data = test_data.cuda()
        test_target = test_target.cuda()

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


def plot_curves(loss, accuracy):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    ax1.plot(loss, color='r')
    ax2.plot(accuracy, color='r')
    plt.tight_layout()
    plt.show()


def plot_classification_maps(predict, target, **kwargs):
    predict_filt = predict.copy()
    predict_filt[target == 0] = 0
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    ax1.set_title('Ground truth')
    ax2.set_title('Predicted map')
    ax3.set_title('Remove background')
    ax1.imshow(target, **kwargs)
    ax2.imshow(predict, **kwargs)
    ax3.imshow(predict_filt, **kwargs)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_dir = './data/Indian_pines_corrected.mat'
    target_dir = './data/Indian_pines_gt.mat'
    prop = 0.2
    batch_size = 2051
    data, target = read_data(data_dir, target_dir)
    train_data, train_target, test_data, test_target = produce_samples(data, target, prop)
    # pdb.set_trace()
    # print('train_data', train_data.shape)
    # print('train_target', train_target.shape)
    # print('test_data', test_data.shape)
    # print('test_traget', test_target.shape)
    # for data in get_one_batch(train_data, train_target, batch_size):
    #     print(data[0].shape, data[1].shape)


