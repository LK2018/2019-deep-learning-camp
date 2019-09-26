# -*- coding: utf-8 -*-

import os
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


def get_masks(target, train_prop, val_prop, save_dir=None):
    assert train_prop + val_prop < 1
    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        train_num = int(round(len(idx) * train_prop))
        val_num = int(round(len(idx) * val_prop))

        np.random.shuffle(idx)
        train_idx = idx[:train_num]
        val_idx = idx[train_num:train_num + val_num]
        test_idx = idx[train_num + val_num:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        folder_name = 'train_' + str(train_prop) + '_val_' + str(val_prop)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'), {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'), {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'), {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


def get_samples(data, target, mask):
    data = data*mask
    target = target*mask

    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T
    target = target.ravel()
    data = data[target != 0]
    target = target[target != 0] - 1

    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).long()

    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    return data, target


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
    mask_save_dir = './data'
    train_prop = 0.2
    val_prop = 0.2
    batch_size = 2051
    data, target = read_data(data_dir, target_dir)
    # train_data, train_target, test_data, test_target = produce_samples(data, target, prop)
    # pdb.set_trace()
    # print('train_data', train_data.shape)
    # print('train_target', train_target.shape)
    # print('test_data', test_data.shape)
    # print('test_traget', test_target.shape)
    # for data in get_one_batch(train_data, train_target, batch_size):
    #     print(data[0].shape, data[1].shape)

    train_mask, val_mask, test_mask = get_masks(target, train_prop, val_prop, save_dir=mask_save_dir)
    train_data, train_target = get_samples(data, target, train_mask)
    pdb.set_trace()

    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
    # ax1.imshow(train_mask)
    # ax2.imshow(val_mask)
    # ax3.imshow(test_mask)
    # ax1.set_title('train mask')
    # ax2.set_title('validation mask')
    # ax3.set_title('test mask')
    # ax4.set_title('complete label')
    # ax4.imshow(train_mask + val_mask + test_mask)
    # plt.tight_layout()
    # plt.show()
    # pdb.set_trace()


