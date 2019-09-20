import matplotlib.pyplot as plt
import torch
from model import BpNet
from tools import *
import pdb


data_dir = './data/Indian_pines_corrected.mat'
target_dir = './data/Indian_pines_gt.mat'
model_dir = './model_save/bpnet_0.2_100_1001.pkl'
data, target = read_data(data_dir, target_dir)
data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T  # data size: [height*width, channel]
data = torch.from_numpy(data).float()

model = BpNet()
model.load_state_dict(torch.load(model_dir))
pred = model(data)
pred = torch.max(pred, 1)[1].data.numpy() + 1
pred = pred.reshape(target.shape[0], target.shape[0])

plot_classification_maps(pred, target, cmap='jet')


