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
class ValidDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        dataset_length = len(os.listdir(root_dir))
        ref_file_name = 'data' + str(dataset_length-1) + '.npy'
        ref_force = np.load(os.path.join(root_dir, ref_file_name), allow_pickle=True, encoding= 'latin1').item()['force'][:6]
        self.ref_force = ref_force
        # print(np.load(os.path.join(root_dir, run, ref_file_name), allow_pickle=True, encoding= 'latin1').item()['GT'])
        # print(ref_force)
        if sum(ref_force) == 0:
            # print('skipping ' + os.path.join(root_dir, run, ref_file_name))
            raise Exception('Reference file has zero force')
        for data in os.listdir(root_dir):
            if data == ref_file_name:
                # print('skipping ' + os.path.join(root_dir, run, data))
                continue
            run_dir = os.path.join(root_dir, data)
            dict_ = np.load(run_dir, allow_pickle=True, encoding= 'latin1').item()
            if sum(dict_['force'][:6]) == 0:
                # print('skipping ' + run_dir)
                continue
            
            dict_['force'] = np.concatenate((ref_force, dict_['force']))
            self.data.append(dict_)
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dict_ = self.data[idx]
        # print(dict_)
        x = dict_['force']
        y = dict_['GT'][:3]*100
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        return x, y
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--dataset', type=str, default='GelDifDataset')
    argparser.add_argument('-m', '--model', type=str, default='gel')
    argparser.add_argument('--directory', '-d', type=str, default='1')
    args = argparser.parse_args()
    model = args.model
    address = args.directory
    address = os.path.join('/media/okemo/extraHDD31/samueljin/data2', address)
    if model != 'gel' and model != 'haptic' and model != 'diff' and model != 'gelhap':
        raise ValueError('Invalid model type')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ValidDataset(address)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    two_pos = [0,0,0]
    print('ref force: ', dataset.ref_force)
    for i, (inputs, targets) in enumerate(data_loader):
        print('new force + pos: ', inputs[0, 6:].numpy())
        targets = targets.numpy()
        full_inputs = inputs.to(device)
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
    two_pos = np.array(two_pos)/len(data_loader)
    print('Two pos: ', two_pos)