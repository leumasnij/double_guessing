import os
import cv2
import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_helpers import Net, HapticDataset, GelResNet, GelSightDataset, GelDifDataset, GelRefDataset, GelHapResNet, RegNet, GelRefResNet, HapNetWithUncertainty, HapDatasetFromTwoPos
from vbllnet import hapVBLLnet
import argparse
argparser = argparse.ArgumentParser()
# argparser.add_argument('--dataset', type=str, default='GelDifDataset')
argparser.add_argument('-m', '--model', type=str, default='gel')
argparser.add_argument('--directory', '-d', type=str, default='/media/okemo/extraHDD31/samueljin/data2')
args = argparser.parse_args()
model = args.model
address = args.directory
if model != 'gel' and model != 'haptic' and model != 'diff' and model != 'gelhap':
    raise ValueError('Invalid model type')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = HapDatasetFromTwoPos(address)
test_size = 50
train_size = len(dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
one_pos = [0,0,0]
two_pos = [0,0,0]
for i, (inputs, targets) in enumerate(test_loader):
    targets = targets.numpy()
    full_inputs = inputs.to(device)
    hap_only = full_inputs[:, 0:6]
    zero = torch.zeros((1,2)).to(device)
    hap_only = torch.cat((hap_only, zero), dim=1).to(device)
    model = hapVBLLnet().to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/vbllnet00_best_model.pth'))
    with torch.no_grad():
        outputs = model(hap_only)
        mean = outputs.predictive.mean.cpu().numpy()
        cov = outputs.predictive.covariance.cpu().numpy()
        error = np.abs(mean[0, 0:3] - targets[0, 0:3])
        std = np.sqrt(np.diag(cov[0, 0:3, 0:3]))
        print('one pos error: ', error)
        print('one pos std: ', std) 
        one_pos += error
    model = hapVBLLnet(input_size=14).to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/vbllnet_2pos_best_model.pth'))
    with torch.no_grad():
        outputs = model(full_inputs)
        mean = outputs.predictive.mean.cpu().numpy()
        cov = outputs.predictive.covariance.cpu().numpy()
        error = np.abs(mean[0, 0:3] - targets[0, 0:3])
        std = np.sqrt(np.diag(cov[0, 0:3, 0:3]))
        print('two pos error: ', error)
        print('two pos std: ', std)
        two_pos += error
    print('-----------------------------------')
one_pos = np.array(one_pos)/test_size
two_pos = np.array(two_pos)/test_size
print('One pos: ', one_pos)
print('Two pos: ', two_pos)

    



# if model == 'gel':
#     model = GelResNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gel_best_model.pth'))
# elif model == 'haptic':
#     model = RegNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/hap_best_model.pth'))
# elif model == 'diff':
#     model = GelResNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/diff_best_model.pth'))
# elif model == 'gelhap':
#     model = GelHapResNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gelhap_best_model.pth'))

