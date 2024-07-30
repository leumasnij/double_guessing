import os
import torch
import vbll
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vbllnet import RegNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import numpy as np
from pyro.infer import Predictive
from PyroNet import BNN_pretrained
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def normalize(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def deterministic_model(weights, deterministic_model, device):
    # deterministic_model = RegNet(input_size=8, output_size=3)
    deterministic_model.fc1.weight = nn.parameter.Parameter(weights['fc1_weight'].mean(0))
    deterministic_model.fc1.bias = nn.parameter.Parameter(weights['fc1_bias'].mean(0))
    deterministic_model.fc2.weight = nn.parameter.Parameter(weights['fc2_weight'].mean(0))
    deterministic_model.fc2.bias = nn.parameter.Parameter(weights['fc2_bias'].mean(0))
    deterministic_model.fc3.weight = nn.parameter.Parameter(weights['fc3_weight'].mean(0))  
    deterministic_model.fc3.bias = nn.parameter.Parameter(weights['fc3_bias'].mean(0))
    deterministic_model.fc4.weight = nn.parameter.Parameter(weights['fc4_weight'].mean(0))
    deterministic_model.fc4.bias = nn.parameter.Parameter(weights['fc4_bias'].mean(0))
    deterministic_model = deterministic_model.to(device)
    deterministic_model.eval()
    return deterministic_model
    

def dataset_helper(folder, model_addr, deterministic_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(model_addr)
    model = BNN_pretrained
    determinstic_model = deterministic_model.to(device)
    dir_list = os.listdir(folder)
    inputs = []
    targets = []
    for file in dir_list:
        file_path = os.path.join(folder, file)
        data_dict = np.load(file_path, allow_pickle=True, encoding= 'latin1').item()
        inputs.append(data_dict['force'])
        targets.append(data_dict['GT'][:3]*100)
        # print(data_dict['GT'])
        
    inputs = torch.tensor(inputs).float()
    inputs = inputs.view(-1, 8)
    targets = torch.tensor(targets).float()
    targets = targets.view(-1, 3)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    pred = Predictive(model=model, posterior_samples=weights)
    outputs = pred(inputs)
    mean = deterministic_model(inputs).cpu().detach().numpy()
    std = outputs['obs'].std(0).cpu().detach().numpy()
    
    return mean, std, inputs.cpu().detach().numpy(), targets.cpu().detach().numpy()


def rand_int_dataset_helper(folder, model_addr, deterministic_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(model_addr)
    model = BNN_pretrained
    determinstic_model = deterministic_model.to(device)
    dir_list = os.listdir(folder)
    inputs = []
    targets = []
    for file in dir_list:
        file_path = os.path.join(folder, file)
        data_dict = np.load(file_path, allow_pickle=True, encoding= 'latin1').item()
        inputs.append(data_dict['force'])
        targets.append(data_dict['GT'][:3]*100)
        # print(data_dict['GT'])
        
    inputs = torch.tensor(inputs).float()
    inputs = inputs.view(-1, 8)
    targets = torch.tensor(targets).float()
    targets = targets.view(-1, 3)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    pred = Predictive(model=model, posterior_samples=weights)
    outputs = pred(inputs)
    mean = deterministic_model(inputs).cpu().detach().numpy()
    std = outputs['obs'].std(0).cpu().detach().numpy()
    
    rand_idx = np.random.permutation(mean.shape[0])
    mean = mean[rand_idx]
    std = std[rand_idx]
    inputs = inputs.cpu().detach().numpy()[rand_idx]
    targets = targets.cpu().detach().numpy()[rand_idx]
    
    return mean, std, inputs, targets


def find_closest_index(source, target, list_of_arrays):

    closest_index = -1
    smallest_distance = float('inf')
    
    # Iterate over the list of arrays
    for i, array in enumerate(list_of_arrays):
        # Compute the average of the source and current array
        avg_array = (source + array) / 2
        # Compute the Euclidean distance to the target array
        distance = np.linalg.norm(avg_array - target)
        # Update the closest index if a smaller distance is found
        if distance < smallest_distance:
            smallest_distance = distance
            closest_index = i
    
    return closest_index

def find_closest_angle(source, list_of_arrays):

    closest_index = -1
    smallest_distance = float('inf')
    
    # Iterate over the list of arrays
    for i, array in enumerate(list_of_arrays):
        # Compute the average of the source and current array
        distance = np.linalg.norm(source - array)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_index = i
    
    return closest_index

class activeDataset(Dataset):
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None, only_zero_init = False):
        super(activeDataset, self).__init__()
        if only_zero_init:
            data_file_name = data_file_name.split('.')[0] + '_zero_init.npy'
        if pre_orgainized:
            file_name = os.path.join(folder, data_file_name)
            self.data = np.load(file_name, allow_pickle=True)
        else:
            self.uncertainty_weight = torch.load(uncertainty_weight_addr)
            self.deter_model = RegNet(input_size=8, output_size=3)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.deter_model = deterministic_model(self.uncertainty_weight, self.deter_model, device)
            self.data = []
            dirlist = os.listdir(folder)
            for run in dirlist:
                if '.' in run:
                    continue
                if len(os.listdir(os.path.join(folder, run))) == 0:
                    continue
                mean, std, inputs, targets = dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, self.deter_model)
                # print(run)
                if only_zero_init:
                    condition = (inputs[:, -2] == 0) & (inputs[:, -1] == 0)
                    indices = np.where(condition)[0]
                    if len(indices) != 1:
                        continue
                    closeset_idx = find_closest_index(mean[indices], targets[indices], mean)
                    GT_ = inputs[closeset_idx][-2:]
                    mean = mean[indices]
                    std = std[indices]
                    inputs = inputs[indices]
                    targets = targets[indices]
                    input_array = np.concatenate([mean, std], axis = 1)
                    
                    dict_ = {'input': input_array, 'GT': GT_}
                    self.data.append(dict_)
                    continue
                
                for i in range(mean.shape[0]):
                    closeset_idx = find_closest_index(mean[i], targets[i], mean)
                    input_array = np.concatenate(np.array([mean[i], std[i]]))
                    # if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                        # print('00_best angle: ' + str(inputs[closeset_idx][-2:]))
                    GT_ = inputs[closeset_idx][-2:]
                    dict_ = {'input': input_array, 'GT': GT_}
                    self.data.append(dict_)
            np.save(os.path.join(folder, data_file_name), self.data)
        print(f'total data: {len(self.data)}') 
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        # print(self.data[idx]['input'], self.data[idx]['GT'])
        return self.data[idx]['input'], self.data[idx]['GT']
            
class GridSearchDataset(Dataset):
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None, only_zero_init = False):
        super(GridSearchDataset, self).__init__()
        if only_zero_init:
            data_file_name = data_file_name.split('.')[0] + '_zero_init.npy'
        if pre_orgainized:
            file_name = os.path.join(folder, data_file_name)
            self.data = np.load(file_name, allow_pickle=True)
        else:
            self.uncertainty_weight = torch.load(uncertainty_weight_addr)
            self.deter_model = RegNet(input_size=8, output_size=3)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.deter_model = deterministic_model(self.uncertainty_weight, self.deter_model, device)
            self.data = []
            dirlist = os.listdir(folder)
            dirlist = np.sort(dirlist)
            for run in dirlist:
                if '.' in run:
                    continue
                if len(os.listdir(os.path.join(folder, run))) == 0:
                    continue
                mean, std, inputs, targets = dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, self.deter_model)
                print(run)
                best_error = 1000
                if only_zero_init:
                    condition = (inputs[:, -2] == 0) & (inputs[:, -1] == 0)
                    indices = np.where(condition)[0]
                    if len(indices) != 1:
                        continue
                    for i in range(mean.shape[0]):
                        input_array = [mean[indices][0], std[indices][0], inputs[i][-2:]]
                        # print(input_array)
                        input_array = np.concatenate(input_array)
                        new_est = (mean[indices] + mean[i])/2
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array, 'GT': L2_error}
                        self.data.append(dict1_)
                        if L2_error < best_error:
                            best_error = L2_error
                    print(f'Best error: {best_error}')
                    continue
                for i in range(mean.shape[0]):
                    for j in range(i+1, mean.shape[0]):
                        input_array1 = [mean[i], std[i], inputs[j][-2:]]
                        input_array1 = np.concatenate(input_array1)
                        input_array2 = [mean[j], std[j], inputs[i][-2:]]
                        input_array2 = np.concatenate(input_array2)
                        # print(input_array)
                        new_est = (mean[j] + mean[i])/2
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array1, 'GT': L2_error}
                        dict2_ = {'input': input_array2, 'GT': L2_error}
                        self.data.append(dict1_)
                        self.data.append(dict2_)
                        if L2_error < best_error:
                            best_error = L2_error
                print(f'Best error: {best_error}')
            
            np.save(os.path.join(folder, data_file_name), self.data)           
        # scores = [obj['GT'] for obj in self.data]
        # mean_score = np.mean(scores)
        # std_score = np.std(scores)  
        # print(f'mean: {mean_score}, std: {std_score}')
        # for obj in self.data:
        #     obj['GT'] = (obj['GT'] - mean_score)/ std_score
        
        print(f'total data: {len(self.data)}')
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['GT']
    
class GridSearchGTMSEDataset(Dataset):
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None, only_zero_init = False):
        super(GridSearchGTMSEDataset, self).__init__()
        if only_zero_init:
            data_file_name = data_file_name.split('.')[0] + '_zero_init.npy'
        if pre_orgainized:
            file_name = os.path.join(folder, data_file_name)
            self.data = np.load(file_name, allow_pickle=True)
        else:
            self.uncertainty_weight = torch.load(uncertainty_weight_addr)
            self.deter_model = RegNet(input_size=8, output_size=3)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.deter_model = deterministic_model(self.uncertainty_weight, self.deter_model, device)
            self.data = []
            dirlist = os.listdir(folder)
            dirlist = np.sort(dirlist)
            for run in dirlist:
                if '.' in run:
                    continue
                if len(os.listdir(os.path.join(folder, run))) == 0:
                    continue
                mean, std, inputs, targets = dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, self.deter_model)
                print(run)
                best_error = 1000
                if only_zero_init:
                    condition = (inputs[:, -2] == 0) & (inputs[:, -1] == 0)
                    indices = np.where(condition)[0]
                    if len(indices) != 1:
                        continue
                    for i in range(mean.shape[0]):
                        error = np.linalg.norm(targets[i] - mean[i])
                        input_array = [mean[indices][0], error, inputs[i][-2:]]
                        # print(input_array)
                        input_array = np.concatenate(input_array)
                        new_est = (mean[indices] + mean[i])/2
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array, 'GT': L2_error}
                        self.data.append(dict1_)
                        if L2_error < best_error:
                            best_error = L2_error
                    print(f'Best error: {best_error}')
                    continue
                for i in range(mean.shape[0]):
                    for j in range(i+1, mean.shape[0]):
                        error1 = targets[i] - mean[i]
                        error2 = targets[j] - mean[j]
                        input_array1 = [mean[i], error1, inputs[j][-2:]]
                        input_array1 = np.concatenate(input_array1)
                        input_array2 = [mean[j], error2, inputs[i][-2:]]
                        input_array2 = np.concatenate(input_array2)
                        # print(input_array)
                        new_est = (mean[j] + mean[i])/2
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array1, 'GT': L2_error}
                        dict2_ = {'input': input_array2, 'GT': L2_error}
                        self.data.append(dict1_)
                        self.data.append(dict2_)
                        if L2_error < best_error:
                            best_error = L2_error
                print(f'Best error: {best_error}')
            
            np.save(os.path.join(folder, data_file_name), self.data)           
        print(f'total data: {len(self.data)}')
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['GT']
    
 
    

class TwoPosGridSearchDataset(Dataset):
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None, two_pos_weight_addr = None, only_zero_init = False):
        super(TwoPosGridSearchDataset, self).__init__()
        self.input_size = 16
        if only_zero_init:
            data_file_name = data_file_name.split('.')[0] + '_zero_init.npy'
            self.input_size = 14
        if pre_orgainized:
            file_name = os.path.join(folder, data_file_name)
            self.data = np.load(file_name, allow_pickle=True)
        else:
            self.uncertainty_weight = torch.load(uncertainty_weight_addr)
            self.deter_model = RegNet(input_size=8, output_size=3)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.deter_model = deterministic_model(self.uncertainty_weight, self.deter_model, device)
            self.two_pos_model = RegNet(input_size=self.input_size, output_size=3)
            self.two_pos_model.load_state_dict(torch.load(two_pos_weight_addr))
            self.two_pos_model = self.two_pos_model.to(device)
            self.data = []
            dirlist = os.listdir(folder)
            dirlist = np.sort(dirlist)
            for run in dirlist:
                if '.' in run:
                    continue
                if len(os.listdir(os.path.join(folder, run))) == 0:
                    continue
                mean, std, inputs, targets = dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, self.deter_model)
                best_error = 1000
                print(run)
                if only_zero_init:
                    condition = (inputs[:, -2] == 0) & (inputs[:, -1] == 0)
                    indices = np.where(condition)[0]
                    if len(indices) != 1:
                        continue
                    for i in range(mean.shape[0]):
                        input_array = [mean[indices][0], std[indices][0], inputs[i][-2:]]
                        # print(input_array)
                        input_array = np.concatenate(input_array)
                        if only_zero_init:
                            new_x = [inputs[indices][0][:6], inputs[i]]
                        else: 
                            new_x = [inputs[indices][0], inputs[i]]
                        new_x = np.concatenate(new_x)
                        new_x = torch.tensor(new_x).float().to(device)
                        new_est = self.two_pos_model(new_x).cpu().detach().numpy()
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array, 'GT': L2_error}
                        self.data.append(dict1_)
                        if L2_error < best_error:
                            best_error = L2_error
                    print(f'Best error: {best_error}')
                    continue
                for i in range(mean.shape[0]):
                    for j in range(i+1, mean.shape[0]):
                        input_array1 = [mean[i], std[i], inputs[j][-2:]]
                        input_array1 = np.concatenate(input_array1)
                        input_array2 = [mean[j], std[j], inputs[i][-2:]]
                        input_array2 = np.concatenate(input_array2)
                        # print(input_array)
                        new_x1 = [inputs[i], inputs[j]]
                        new_x2 = [inputs[j], inputs[i]]
                        new_x1 = np.concatenate(new_x1)
                        new_x2 = np.concatenate(new_x2)
                        new_est1 = self.two_pos_model(torch.tensor(new_x1).float().to(device)).cpu().detach().numpy()
                        new_est2 = self.two_pos_model(torch.tensor(new_x2).float().to(device)).cpu().detach().numpy()
                        new_est = (new_est1 + new_est2)/2
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array1, 'GT': L2_error}
                        dict2_ = {'input': input_array2, 'GT': L2_error}
                        self.data.append(dict1_)
                        self.data.append(dict2_)
                        if L2_error < best_error:
                            best_error = L2_error
                print(f'Best error: {best_error}')
            
            np.save(os.path.join(folder, data_file_name), self.data)           
        print(f'total data: {len(self.data)}')
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['GT']
    
    
    
class TwoPosGridSearchSparseDataset(Dataset):
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None, two_pos_weight_addr = None, only_zero_init = False):
        super(TwoPosGridSearchSparseDataset, self).__init__()
        self.input_size = 16
        if only_zero_init:
            data_file_name = data_file_name.split('.')[0] + '_zero_init.npy'
            self.input_size = 14
        if pre_orgainized:
            file_name = os.path.join(folder, data_file_name)
            self.data = np.load(file_name, allow_pickle=True)
        else:
            self.uncertainty_weight = torch.load(uncertainty_weight_addr)
            self.deter_model = RegNet(input_size=8, output_size=3)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.deter_model = deterministic_model(self.uncertainty_weight, self.deter_model, device)
            self.two_pos_model = RegNet(input_size=self.input_size, output_size=3)
            self.two_pos_model.load_state_dict(torch.load(two_pos_weight_addr))
            self.two_pos_model = self.two_pos_model.to(device)
            self.data = []
            dirlist = os.listdir(folder)
            dirlist = np.sort(dirlist)
            for run in dirlist:
                if '.' in run:
                    continue
                if len(os.listdir(os.path.join(folder, run))) == 0:
                    continue
                mean, std, inputs, targets = rand_int_dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, self.deter_model)
                best_error = 1000
                print(run)
                if only_zero_init:
                    condition = (inputs[:, -2] == 0) & (inputs[:, -1] == 0)
                    indices = np.where(condition)[0]
                    if len(indices) != 1:
                        continue
                    for i in range(int(mean.shape[0]/10)):
                        input_array = [mean[indices][0], std[indices][0], inputs[i][-2:]]
                        # print(input_array)
                        input_array = np.concatenate(input_array)
                        if only_zero_init:
                            new_x = [inputs[indices][0][:6], inputs[i]]
                        else:
                            new_x = [inputs[indices][0], inputs[i]]
                        new_x = np.concatenate(new_x)
                        new_x = torch.tensor(new_x).float().to(device)
                        new_est = self.two_pos_model(new_x).cpu().detach().numpy()
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array, 'GT': L2_error}
                        self.data.append(dict1_)
                        if L2_error < best_error:
                            best_error = L2_error
                    print(f'Best error: {best_error}')
                    continue
                for i in range(int(mean.shape[0]/10)):
                    for j in range(i+1, int(mean.shape[0]/10)):
                        input_array1 = [mean[i], std[i], inputs[j][-2:]]
                        input_array1 = np.concatenate(input_array1)
                        input_array2 = [mean[j], std[j], inputs[i][-2:]]
                        input_array2 = np.concatenate(input_array2)
                        # print(input_array)
                        new_x1 = [inputs[i], inputs[j]]
                        new_x2 = [inputs[j], inputs[i]]
                        new_x1 = np.concatenate(new_x1)
                        new_x2 = np.concatenate(new_x2)
                        new_est1 = self.two_pos_model(torch.tensor(new_x1).float().to(device)).cpu().detach().numpy()
                        new_est2 = self.two_pos_model(torch.tensor(new_x2).float().to(device)).cpu().detach().numpy()
                        new_est = (new_est1 + new_est2)/2
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array1, 'GT': L2_error}
                        dict2_ = {'input': input_array2, 'GT': L2_error}
                        self.data.append(dict1_)
                        self.data.append(dict2_)
                        if L2_error < best_error:
                            best_error = L2_error
                print(f'Best error: {best_error}')
            
            np.save(os.path.join(folder, data_file_name), self.data)           
        print(f'total data: {len(self.data)}')
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['GT']


class activeNet(nn.Module):
    def __init__(self):
        super(activeNet, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x   
    
class GridSearchNet(nn.Module):
    def __init__(self):
        super(GridSearchNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

class DenserNet(nn.Module):
    def __init__(self):
        super(DenserNet, self).__init__()
        self.fc1 = nn.Linear(8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x 
    
    
class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # Decoder
        self.up_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.up_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        self.output_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        x = x.view(-1, 1, 8)
        x1 = F.relu(self.enc_conv1(x))
        x1_pooled = self.pool(x1)
        x2 = F.relu(self.enc_conv2(x1_pooled))
        x2_pooled = self.pool(x2)
        x3 = F.relu(self.enc_conv3(x2_pooled))
        # Decoding path
        x3_up = self.up_conv1(x3)
        x3_up = torch.cat([x3_up, x2], dim=1)  # Skip connection
        x4 = F.relu(self.dec_conv1(x3_up))
        x4_up = self.up_conv2(x4)
        x4_up = torch.cat([x4_up, x1], dim=1)  # Skip connection
        x5 = F.relu(self.dec_conv2(x4_up))
        # Output layer
        out = self.output_conv(x5)
        # Flatten the output to match the target shape (batch_size, 1)
        
        return out.sum(dim=2).view(-1, 1)
    
def train_activenet(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    lowest_loss = 1000
    model = model.to(device)
    for epoch in range(epochs):  # Number of epochs
        cum_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                optimizer.zero_grad()
                inputs = inputs.to(device)
                if len(targets.shape) == 1:
                    targets = targets.view(-1, 1)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
                pbar.set_postfix({'loss': cum_loss/(i+1)})
                pbar.update(1)
        val_loss = 0
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (inputs, targets) in enumerate(val_loader):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    if len(targets.shape) == 1:
                        targets = targets.view(-1, 1)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    pbar.set_postfix({'loss': val_loss/(i+1)})
                    pbar.update(1)
        val_loss /= len(val_loader)
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(model.state_dict(), save_path + '_best_model.pth')
        torch.save(model.state_dict(), save_path + '_model.pth')
        print(f"Epoch {epoch+1}/{epochs}, loss: {cum_loss/len(train_loader)}, val_loss: {val_loss}")
        
        
def test_active(active_model, uncertainty_model_adr, active_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    dirlist = os.listdir(data_folder)
    error_before_active = [0,0,0]
    error_after_active = [0,0,0]
    improve_count = 0
    GT = []
    pred1 = []
    pred2 = []
    dirlist = np.random.permutation(dirlist)
    dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        print(run)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                
                input_array = np.concatenate(np.array([mean[i], std[i]]))
                input_array = torch.tensor(input_array).float().to(device)
                output = active_model(input_array)
                closest_idx = find_closest_angle(output.cpu().detach().numpy(), inputs[:, -2:])
                new_est = (mean[closest_idx] + mean[i])/2
                error_without_active = targets[i] - mean[i]
                error_with_active = targets[i] - new_est
                print(f'Error without active: {error_without_active}')
                print(f'Error with active: {error_with_active}')
                print('-----------------------------------')
                error_before_active += np.abs(error_without_active)
                error_after_active += np.abs(error_with_active)
                pred1.append(mean[i])
                pred2.append(new_est)
                GT.append(targets[i])
                if np.linalg.norm(error_with_active) < np.linalg.norm(error_without_active):
                    improve_count += 1
    print(f'Error before active: {error_before_active/len(dirlist)}')
    print(f'Error after active: {error_after_active/len(dirlist)}')
    import matplotlib.pyplot as plt
    plt.figure()
    total_error_without_active = np.linalg.norm(np.array(pred1) - np.array(GT), axis = 1)
    total_error_with_active = np.linalg.norm(np.array(pred2) - np.array(GT), axis = 1)
    
    plt.figure()
    plt.subplot(2,2,1)
    refer = np.linspace(-10, 10, 100)
    # graph GT versus pred for each axis
    plt.scatter(np.array(GT)[:,0], np.array(pred1)[:,0], label = 'Without Active')
    plt.scatter(np.array(GT)[:,0], np.array(pred2)[:,0], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('X axis')
    
    plt.subplot(2,2,2)
    plt.scatter(np.array(GT)[:,1], np.array(pred1)[:,1], label = 'Without Active')
    plt.scatter(np.array(GT)[:,1], np.array(pred2)[:,1], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Y axis')
    
    plt.subplot(2,2,3)
    plt.scatter(np.array(GT)[:,2], np.array(pred1)[:,2], label = 'Without Active')
    plt.scatter(np.array(GT)[:,2], np.array(pred2)[:,2], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Z axis')
    
    plt.subplot(2,2,4)
    # For last plot, plot the total error for both methods
    plt.plot(total_error_without_active, label = 'Without Active')
    plt.plot(total_error_with_active, label = 'With Active')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Error') 
    plt.title('Total Error')
    save_name = active_model_adr.split('/')[-1].split('.')[0] + '_error_plot.png'
    print(f'Improvement count: {improve_count}')
    print(f'Improvement rate: {improve_count/len(GT)}')
    print('save_name: ', save_name)
    plt.savefig(save_name)
    
def grid_search(model, device, inputs, grid_size = 100):
    # model = model.to(device)
    angle2 = np.linspace(np.pi/6, 5*np.pi/6, grid_size)
    angle1 = np.linspace(-np.pi/2, np.pi/2, grid_size)
    output_mat = np.ones((angle1.shape[0], angle2.shape[0]))
    best_angle = [0,0]
    for i in range(angle1.shape[0]):
        for j in range(angle2.shape[0]):
            input_array = [inputs, [angle1[i], angle2[j]]]
            input_array = np.concatenate(input_array)
            out = model(torch.tensor(input_array).float().to(device))
            if out.shape[0] != 1:
                out = out.squeeze(0)
            output_mat[i, j] = out
    best_angle = np.argmin(output_mat)
    print(best_angle)
    return_angle1, return_angle2 = np.unravel_index(np.argmin(output_mat, axis=None), output_mat.shape)
    return angle1[return_angle1], angle2[return_angle2]
            
            
            

def test_grid_search(active_model, uncertainty_model_adr, active_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    dirlist = os.listdir(data_folder)
    error_before_active = [0,0,0]
    error_after_active = [0,0,0]
    improve_count = 0
    GT = []
    pred1 = []
    pred2 = []
    dirlist = np.random.permutation(dirlist)
    # dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        print(run)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                
                input_array = [mean[i], std[i]]
                input_array = np.concatenate(input_array)
                new_angle1, new_angle2 = grid_search(active_model, device, input_array)
                print(f'New angle1: {new_angle1}, New angle2: {new_angle2}')
                closest_idx = find_closest_angle([new_angle1, new_angle2], inputs[:, -2:])
                new_est = (mean[closest_idx] + mean[i])/2
                error_without_active = targets[i] - mean[i]
                error_with_active = targets[i] - new_est
                print(f'Error without active: {error_without_active}')
                print(f'Error with active: {error_with_active}')
                print('-----------------------------------')
                error_before_active += np.abs(error_without_active)
                error_after_active += np.abs(error_with_active)
                pred1.append(mean[i])
                pred2.append(new_est)
                GT.append(targets[i])
                if np.linalg.norm(error_with_active) < np.linalg.norm(error_without_active):
                    improve_count += 1
    print(f'Error before active: {error_before_active/len(GT)}')
    print(f'Error after active: {error_after_active/len(GT)}')
    import matplotlib.pyplot as plt
    plt.figure()
    total_error_without_active = np.linalg.norm(np.array(pred1) - np.array(GT), axis = 1)
    total_error_with_active = np.linalg.norm(np.array(pred2) - np.array(GT), axis = 1)
    
    plt.figure()
    plt.subplot(2,2,1)
    refer = np.linspace(-10, 10, 100)
    # graph GT versus pred for each axis
    plt.scatter(np.array(GT)[:,0], np.array(pred1)[:,0], label = 'Without Active')
    plt.scatter(np.array(GT)[:,0], np.array(pred2)[:,0], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('X axis')
    
    plt.subplot(2,2,2)
    plt.scatter(np.array(GT)[:,1], np.array(pred1)[:,1], label = 'Without Active')
    plt.scatter(np.array(GT)[:,1], np.array(pred2)[:,1], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Y axis')
    
    plt.subplot(2,2,3)
    plt.scatter(np.array(GT)[:,2], np.array(pred1)[:,2], label = 'Without Active')
    plt.scatter(np.array(GT)[:,2], np.array(pred2)[:,2], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Z axis')
    
    plt.subplot(2,2,4)
    # For last plot, plot the total error for both methods
    plt.plot(total_error_without_active, label = 'Without Active')
    plt.plot(total_error_with_active, label = 'With Active')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Error') 
    plt.title('Total Error')
    save_name = active_model_adr.split('/')[-1].split('.')[0] + '_error_plot.png'
    print(f'Improvement count: {improve_count}')
    print(f'Improvement rate: {improve_count/len(GT)}')
    print('save_name: ', save_name)
    plt.savefig(save_name)
    result = [error_before_active/len(GT), error_after_active/len(GT), improve_count/len(GT)]
    result = np.array(result).reshape(1,3).astype(np.float64)
    np.savetxt(save_name.split('.')[0] + '_error.txt', result, fmt='%.18e', newline=' ')

def test_grid_with_MSEGT_search(active_model, uncertainty_model_adr, active_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    dirlist = os.listdir(data_folder)
    error_before_active = [0,0,0]
    error_after_active = [0,0,0]
    improve_count = 0
    GT = []
    pred1 = []
    pred2 = []
    dirlist = np.random.permutation(dirlist)
    # dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        print(run)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                
                error = targets[i] - mean[i]
                input_array = [mean[i], error]
                input_array = np.concatenate(input_array)
                new_angle1, new_angle2 = grid_search(active_model, device, input_array)
                print(f'New angle1: {new_angle1}, New angle2: {new_angle2}')
                closest_idx = find_closest_angle([new_angle1, new_angle2], inputs[:, -2:])
                new_est = (mean[closest_idx] + mean[i])/2
                error_without_active = targets[i] - mean[i]
                error_with_active = targets[i] - new_est
                print(f'Error without active: {error_without_active}')
                print(f'Error with active: {error_with_active}')
                print('-----------------------------------')
                error_before_active += np.abs(error_without_active)
                error_after_active += np.abs(error_with_active)
                pred1.append(mean[i])
                pred2.append(new_est)
                GT.append(targets[i])
                if np.linalg.norm(error_with_active) < np.linalg.norm(error_without_active):
                    improve_count += 1
    print(f'Error before active: {error_before_active/len(GT)}')
    print(f'Error after active: {error_after_active/len(GT)}')
    import matplotlib.pyplot as plt
    plt.figure()
    total_error_without_active = np.linalg.norm(np.array(pred1) - np.array(GT), axis = 1)
    total_error_with_active = np.linalg.norm(np.array(pred2) - np.array(GT), axis = 1)
    
    plt.figure()
    plt.subplot(2,2,1)
    refer = np.linspace(-10, 10, 100)
    # graph GT versus pred for each axis
    plt.scatter(np.array(GT)[:,0], np.array(pred1)[:,0], label = 'Without Active')
    plt.scatter(np.array(GT)[:,0], np.array(pred2)[:,0], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('X axis')
    
    plt.subplot(2,2,2)
    plt.scatter(np.array(GT)[:,1], np.array(pred1)[:,1], label = 'Without Active')
    plt.scatter(np.array(GT)[:,1], np.array(pred2)[:,1], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Y axis')
    
    plt.subplot(2,2,3)
    plt.scatter(np.array(GT)[:,2], np.array(pred1)[:,2], label = 'Without Active')
    plt.scatter(np.array(GT)[:,2], np.array(pred2)[:,2], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Z axis')
    
    plt.subplot(2,2,4)
    # For last plot, plot the total error for both methods
    plt.plot(total_error_without_active, label = 'Without Active')
    plt.plot(total_error_with_active, label = 'With Active')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Error') 
    plt.title('Total Error')
    save_name = active_model_adr.split('/')[-1].split('.')[0] + '_error_plot.png'
    print(f'Improvement count: {improve_count}')
    print(f'Improvement rate: {improve_count/len(GT)}')
    print('save_name: ', save_name)
    plt.savefig(save_name)
    result = np.array([error_before_active/len(GT), error_after_active/len(GT), improve_count/len(GT)]) 
    np.savetxt(save_name.split('.')[0] + '_error.txt', result, fmt='%.18e', newline=' ')


def test_two_pos_grid_search(active_model, uncertainty_model_adr, active_model_adr, two_pos_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    two_pos_model = torch.load(two_pos_model_adr)
    model_input_size = two_pos_model['fc1.weight'].shape[1]
    two_pos_model = RegNet(input_size=model_input_size, output_size=3)
    two_pos_model.load_state_dict(torch.load(two_pos_model_adr))
    two_pos_model = two_pos_model.to(device)
    
    dirlist = os.listdir(data_folder)
    error_before_active = [0,0,0]
    error_after_active = [0,0,0]
    improve_count = 0
    GT = []
    pred1 = []
    pred2 = []
    save_dict = {}
    dirlist = np.random.permutation(dirlist)
    # dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        print(run)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                
                input_array = [mean[i], std[i]]
                input_array = np.concatenate(input_array)
                new_angle1, new_angle2 = grid_search(active_model, device, input_array)
                # print(f'New angle1: {new_angle1}, New angle2: {new_angle2}')
                closest_idx = find_closest_angle([new_angle1, new_angle2], inputs[:, -2:])
                # print(f'new angle estimate: {[new_angle1, new_angle2]}')
                # print(f'new angle actual: {inputs[closest_idx][-2:]}')
                random_idx = np.random.randint(0, mean.shape[0])
                if random_idx == closest_idx:
                    random_idx = (random_idx + 1) % mean.shape[0]
                save_dict[run] = inputs[closest_idx][-2:]
                
                if model_input_size == 14:
                    new_x = [inputs[i][:6], inputs[closest_idx]]
                    new_x = np.concatenate(new_x)
                    new_x2 = [inputs[i][:6], inputs[random_idx]]
                    new_x2 = np.concatenate(new_x2)
                    input = [new_x, new_x2]
                    input = torch.tensor(input).float().to(device)
                else:
                    new_x = [inputs[i], inputs[closest_idx]]
                    new_x = np.concatenate(new_x)
                    new_x2 = [inputs[i], inputs[random_idx]]
                    new_x2 = np.concatenate(new_x2)
                    input = [new_x, new_x2]
                    input = torch.tensor(input).float().to(device)
                with torch.no_grad():   
                    two_pos_model.eval()
                    new_est = two_pos_model(input).cpu().detach().numpy()
                error_without_active = targets[i] - new_est[1]
                error_with_active = targets[i] - new_est[0]
                print(f'Error without active: {error_without_active}')
                print(f'Error with active: {error_with_active}')
                print('-----------------------------------')
                error_before_active = np.add(error_before_active, np.abs(error_without_active))
                error_after_active = np.add(error_after_active, np.abs(error_with_active))
                pred1.append(new_est[1])
                pred2.append(new_est[0])
                GT.append(targets[i])
                if np.linalg.norm(error_with_active) < np.linalg.norm(error_without_active):
                    improve_count += 1
    print(f'Error before active: {error_before_active/len(GT)}')
    print(f'Error after active: {error_after_active/len(GT)}')
    import matplotlib.pyplot as plt
    plt.figure()
    total_error_without_active = np.linalg.norm(np.array(pred1) - np.array(GT), axis = 1)
    total_error_with_active = np.linalg.norm(np.array(pred2) - np.array(GT), axis = 1)
    
    plt.figure()
    plt.subplot(2,2,1)
    refer = np.linspace(-10, 10, 100)
    # graph GT versus pred for each axis
    plt.scatter(np.array(GT)[:,0], np.array(pred1)[:,0], label = 'Without Active')
    plt.scatter(np.array(GT)[:,0], np.array(pred2)[:,0], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('X axis')
    
    plt.subplot(2,2,2)
    plt.scatter(np.array(GT)[:,1], np.array(pred1)[:,1], label = 'Without Active')
    plt.scatter(np.array(GT)[:,1], np.array(pred2)[:,1], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Y axis')
    
    plt.subplot(2,2,3)
    plt.scatter(np.array(GT)[:,2], np.array(pred1)[:,2], label = 'Without Active')
    plt.scatter(np.array(GT)[:,2], np.array(pred2)[:,2], label = 'With Active')
    plt.plot(refer, refer, 'r--')
    plt.legend()
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('Z axis')
    
    plt.subplot(2,2,4)
    # For last plot, plot the total error for both methods
    plt.plot(total_error_without_active, label = 'Without Active')
    plt.plot(total_error_with_active, label = 'With Active')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Error') 
    plt.title('Total Error')
    save_name = active_model_adr.split('/')[-1].split('.')[0] + '_error_plot.png'
    print(f'Improvement count: {improve_count}')
    print(f'Improvement rate: {improve_count/len(GT)}')
    print('save_name: ', save_name)
    plt.savefig(save_name)
    
    np.save(save_name.split('.')[0] + '.npy', save_dict)
    
    result = np.array([error_before_active/len(GT), error_after_active/len(GT), improve_count/len(GT)]) 
    np.savetxt(save_name.split('.')[0] + '_error.txt', result, fmt='%.18e', newline=' ')
    
def test_joint_model_infer_accuracy(active_model, uncertainty_model_adr, active_model_adr, two_pos_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    two_pos_model = torch.load(two_pos_model_adr)
    model_input_size = two_pos_model['fc1.weight'].shape[1]
    two_pos_model = RegNet(input_size=model_input_size, output_size=3)
    two_pos_model.load_state_dict(torch.load(two_pos_model_adr))
    two_pos_model = two_pos_model.to(device)
    
    dirlist = os.listdir(data_folder)
    best_match_infer_count = 0
    best_error_greater_count = 0
    best_error_fewer_count = 0
    dirlist = np.random.permutation(dirlist)
    # dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        print(run)
        # print(targets)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                input_act_array = []
                input_2pos_array = []
                for j in range(mean.shape[0]):
                    act_input = [mean[i], std[i] ,inputs[j][-2:]]
                    if model_input_size == 14:
                        Twopos_input = [inputs[i][:6], inputs[j]]
                    else:
                        Twopos_input = [inputs[i], inputs[j]]
                    act_input = np.concatenate(act_input)
                    Twopos_input = np.concatenate(Twopos_input)
                    input_act_array.append(act_input)
                    input_2pos_array.append(Twopos_input)
                input_act_array = np.array(input_act_array)
                input_2pos_array = np.array(input_2pos_array)
                input_act_array = torch.tensor(input_act_array).float().to(device)
                input_2pos_array = torch.tensor(input_2pos_array).float().to(device)
                # with torch.no_grad():
                #     active_model.eval()
                #     two_pos_model.eval()
                #     act_output = active_model(input_act_array).cpu().detach().numpy()
                #     two_pos_output = two_pos_model(input_2pos_array).cpu().detach().numpy()
                # for j in range(mean.shape[0]):
                #     two_pos_error = np.linalg.norm(targets[j] - two_pos_output[j])
                #     print(f'Two pos error: {two_pos_error}')
                #     print(f'Active error: {act_output[j]}')
                #     diff = np.abs(two_pos_error - act_output[j])
                #     print(f'Diff: {diff}')
                input_array = [mean[i], std[i]]
                input_array = np.concatenate(input_array)
                inferred_angle = grid_search(active_model, device, input_array)
                print(f'Inferred angle: {inferred_angle}')  
                close_idx = find_closest_angle(inferred_angle, inputs[:, -2:])
                close_angle = inputs[close_idx][-2:]
                print(f'Close angle: {close_angle}')
                with torch.no_grad():
                    two_pos_output = two_pos_model(input_2pos_array).cpu().detach().numpy()
                    act_output = active_model(input_act_array).cpu().detach().numpy()
                    
                total_error = np.linalg.norm(targets - two_pos_output, axis = 1)
                best_idx = np.argmin(total_error)
                best_angle = inputs[best_idx][-2:]
                print(f'Best angle: {best_angle}')
                best_angle_error = np.linalg.norm(targets[best_idx] - two_pos_output[best_idx])
                best_angle_inf_error = act_output[best_idx]
                inferred_angle_error = np.linalg.norm(targets[close_idx] - two_pos_output[close_idx])
                inferred_angle_inf_error = act_output[close_idx]
                print(f'Best angle error: {best_angle_error}')
                print(f'Best angle inf error: {best_angle_inf_error}')
                print(f'Inferred angle error: {inferred_angle_error}')
                print(f'Inferred angle inf error: {inferred_angle_inf_error}')
                if best_angle[0] == close_angle[0] and best_angle[1] == close_angle[1]:
                    best_match_infer_count += 1
                else :
                    if best_angle_inf_error < inferred_angle_inf_error:
                        best_error_fewer_count += 1
                    else:
                        best_error_greater_count += 1
    print(f'Best match infer count: {best_match_infer_count}')
    print(f'Best error fewer count: {best_error_fewer_count}')
    print(f'Best error greater count: {best_error_greater_count}')
    print(f'Model: {active_model_adr}')
            
def test_act_model_infer_accuracy(active_model, uncertainty_model_adr, active_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    
    dirlist = os.listdir(data_folder)
    best_match_infer_count = 0
    best_error_greater_count = 0
    best_error_fewer_count = 0
    save_dict = {}
    dirlist = np.random.permutation(dirlist)
    # dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        print(run)
        # print(targets)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                input_act_array = []
                for j in range(mean.shape[0]):
                    act_input = [mean[i], std[i] ,inputs[j][-2:]]
                    act_input = np.concatenate(act_input)
                    input_act_array.append(act_input)
                input_act_array = np.array(input_act_array)
                input_act_array = torch.tensor(input_act_array).float().to(device)
                # with torch.no_grad():
                #     active_model.eval()
                #     two_pos_model.eval()
                #     act_output = active_model(input_act_array).cpu().detach().numpy()
                #     two_pos_output = two_pos_model(input_2pos_array).cpu().detach().numpy()
                # for j in range(mean.shape[0]):
                #     two_pos_error = np.linalg.norm(targets[j] - two_pos_output[j])
                #     print(f'Two pos error: {two_pos_error}')
                #     print(f'Active error: {act_output[j]}')
                #     diff = np.abs(two_pos_error - act_output[j])
                #     print(f'Diff: {diff}')
                input_array = [mean[i], std[i]]
                input_array = np.concatenate(input_array)
                inferred_angle = grid_search(active_model, device, input_array)
                print(f'Inferred angle: {inferred_angle}')  
                close_idx = find_closest_angle(inferred_angle, inputs[:, -2:])
                close_angle = inputs[close_idx][-2:]
                print(f'Close angle: {close_angle}')
                with torch.no_grad():
                    act_output = active_model(input_act_array).cpu().detach().numpy()
                    
                best_idx = find_closest_index(mean[i], targets[i], mean)
                best_angle = inputs[best_idx][-2:]
                print(f'Best angle: {best_angle}')
                best_est = (mean[best_idx] + mean[i])/2
                best_angle_error = np.linalg.norm(targets[best_idx] - best_est)
                best_angle_inf_error = act_output[best_idx]
                infer_est = (mean[close_idx] + mean[i])/2
                inferred_angle_error = np.linalg.norm(targets[close_idx] - infer_est)
                inferred_angle_inf_error = act_output[close_idx]
                print(f'Best angle error: {best_angle_error}')
                print(f'Best angle inf error: {best_angle_inf_error}')
                print(f'Inferred angle error: {inferred_angle_error}')
                print(f'Inferred angle inf error: {inferred_angle_inf_error}')
                if best_angle[0] == close_angle[0] and best_angle[1] == close_angle[1]:
                    best_match_infer_count += 1
                else :
                    if best_angle_inf_error < inferred_angle_inf_error:
                        best_error_fewer_count += 1
                    else:
                        best_error_greater_count += 1
    print(f'Best match infer count: {best_match_infer_count}')
    print(f'Best error fewer count: {best_error_fewer_count}')
    print(f'Best error greater count: {best_error_greater_count}')
    print(f'Model: {active_model_adr}')

def test_aactMSE_model_infer_accuracy(active_model, uncertainty_model_adr, active_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model.load_state_dict(torch.load(active_model_adr))
    active_model = active_model.to(device)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    
    dirlist = os.listdir(data_folder)
    best_match_infer_count = 0
    best_error_greater_count = 0
    best_error_fewer_count = 0
    dirlist = np.random.permutation(dirlist)
    # dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(data_folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(data_folder, run), uncertainty_model_adr, deter_model)
        error = targets - mean
        print(run)
        # print(targets)
        for i in range(mean.shape[0]):
            if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                input_act_array = []
                for j in range(mean.shape[0]):
                    act_input = [mean[i], error[i] ,inputs[j][-2:]]
                    act_input = np.concatenate(act_input)
                    input_act_array.append(act_input)
                input_act_array = np.array(input_act_array)
                input_act_array = torch.tensor(input_act_array).float().to(device)
                # with torch.no_grad():
                #     active_model.eval()
                #     two_pos_model.eval()
                #     act_output = active_model(input_act_array).cpu().detach().numpy()
                #     two_pos_output = two_pos_model(input_2pos_array).cpu().detach().numpy()
                # for j in range(mean.shape[0]):
                #     two_pos_error = np.linalg.norm(targets[j] - two_pos_output[j])
                #     print(f'Two pos error: {two_pos_error}')
                #     print(f'Active error: {act_output[j]}')
                #     diff = np.abs(two_pos_error - act_output[j])
                #     print(f'Diff: {diff}')
                input_array = [mean[i], error[i]]
                input_array = np.concatenate(input_array)
                inferred_angle = grid_search(active_model, device, input_array)
                print(f'Inferred angle: {inferred_angle}')  
                close_idx = find_closest_angle(inferred_angle, inputs[:, -2:])
                close_angle = inputs[close_idx][-2:]
                print(f'Close angle: {close_angle}')
                with torch.no_grad():
                    act_output = active_model(input_act_array).cpu().detach().numpy()
                    
                best_idx = find_closest_index(mean[i], targets[i], mean)
                best_angle = inputs[best_idx][-2:]
                print(f'Best angle: {best_angle}')
                best_est = (mean[best_idx] + mean[i])/2
                best_angle_error = np.linalg.norm(targets[best_idx] - best_est)
                best_angle_inf_error = act_output[best_idx]
                infer_est = (mean[close_idx] + mean[i])/2
                inferred_angle_error = np.linalg.norm(targets[close_idx] - infer_est)
                inferred_angle_inf_error = act_output[close_idx]
                print(f'Best angle error: {best_angle_error}')
                print(f'Best angle inf error: {best_angle_inf_error}')
                print(f'Inferred angle error: {inferred_angle_error}')
                print(f'Inferred angle inf error: {inferred_angle_inf_error}')
                if best_angle[0] == close_angle[0] and best_angle[1] == close_angle[1]:
                    best_match_infer_count += 1
                else :
                    if best_angle_inf_error < inferred_angle_inf_error:
                        best_error_fewer_count += 1
                    else:
                        best_error_greater_count += 1
    print(f'Best match infer count: {best_match_infer_count}')
    print(f'Best error fewer count: {best_error_fewer_count}')
    print(f'Best error greater count: {best_error_greater_count}')
    print(f'Model: {active_model_adr}')
 
    
def graph_angle_matching(uncertainty_weight_addr, device, folder):
    uncertainty_weight = torch.load(uncertainty_weight_addr)
    deter_model = RegNet(input_size=8, output_size=3)
    deter_model = deterministic_model(uncertainty_weight, deter_model, device)
    dirlist = os.listdir(folder)
    np.random.seed(0)
    dirlist = np.random.permutation(dirlist)
    dirlist = dirlist[:100]
    for run in dirlist:
        if '.' in run:
            continue
        if len(os.listdir(os.path.join(folder, run))) == 0:
            continue
        mean, std, inputs, targets = dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, deter_model)
        print(run)
        initial_angles = []
        final_angles = []
        for i in range(mean.shape[0]):
            closeset_idx = find_closest_index(mean[i], targets[i], mean)
            GT_ = inputs[closeset_idx][-2:]
            initial_angles.append(inputs[i][-2:])
            final_angles.append(GT_)
    
        initial_angles = np.array(initial_angles)
        final_angles = np.array(final_angles) 
        save_dir = 'data/' + run + '.npy'
        np.save(save_dir, {'initial_angles': initial_angles, 'final_angles': final_angles})
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # scatter = ax.scatter(initial_angles[:,0], initial_angles[:,1], final_angles[:,0], c='r', marker='o')
        # ax.set_xlabel('initial angle1')
        # ax.set_ylabel('initial angle2')
        # ax.set_zlabel('final angle1')
        # ax.set_title('3D Scatter Plot')
        # fig.colorbar(scatter)
        # plt.savefig('fig/' + run + 'angle1.png')
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # scatter = ax.scatter(initial_angles[:,0], initial_angles[:,1], final_angles[:,1], c='r', marker='o')
        # ax.set_xlabel('initial angle1')
        # ax.set_ylabel('initial angle2')
        # ax.set_zlabel('final angle2')
        # ax.set_title('3D Scatter Plot')
        # fig.colorbar(scatter)
        # plt.savefig('fig/' + run + 'angle2.png')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GSNet', help='Model to use, chose from GSNet, ActiveNet, TwoPosGridSearch, TwoPosGridSearchAll, GSMSEnet, TwoPosGridSearchSparse')
    parser.add_argument('--pre_orgainized', '-po', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--zero_init', '-zi', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--train', '-t', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--test', '-te', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--infer', '-i', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--batch_size', '-bs', type=int, default=48)
    parser.add_argument('--epochs', '-e', type=int, default=500)
    
    args = parser.parse_args()
    model_name = args.model
    pre_orgainized = bool(args.pre_orgainized)
    zero_init = bool(args.zero_init)
    train = bool(args.train)
    test = bool(args.test)
    infer = bool(args.infer)
    batch_size = args.batch_size
    epochs = args.epochs
    
    print(f'pre_orgainized: {pre_orgainized}')
    folder = '/media/okemo/extraHDD31/samueljin/data2'
    uncertainty_weight_addr = '/media/okemo/extraHDD31/samueljin/Model/bnn1_best_model.pth'
    twoPos_weight_addr = None
    if model_name == 'GSNet':
        model = GridSearchNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/GSNet1'
        data_file_name = 'grid_search_dataset1.npy'
        test_model = test_grid_search
        inference_model = test_act_model_infer_accuracy
        if train:
            dataset = GridSearchDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, only_zero_init = zero_init)
    elif model_name == 'GSUNet':
        model = UNet1D()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/GSUNet1'
        data_file_name = 'grid_search_dataset1.npy'
        test_model = test_grid_search
        inference_model = test_act_model_infer_accuracy
        if train:
            dataset = GridSearchDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, only_zero_init = zero_init)
    elif model_name == 'GSDenseNet':
        model = DenserNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/GSDenseNet1'
        data_file_name = 'grid_search_dataset1.npy'
        test_model = test_grid_search
        inference_model = test_act_model_infer_accuracy
        if train:
            dataset = GridSearchDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, only_zero_init = zero_init)
    elif model_name == 'ActiveNet':
        model = activeNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/ActiveNet1'
        data_file_name = 'active_dataset1.npy'
        test_model = test_active
        if train:
            dataset = activeDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, only_zero_init = zero_init)
    elif model_name == 'TwoPosGridSearch':
        model = GridSearchNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/TwoPosGridSearch1'
        data_file_name = '2pgs_dataset1.npy'
        twoPos_weight_addr = '/media/okemo/extraHDD31/samueljin/Model/MLP2Pos_best_model.pth'
        test_model = test_two_pos_grid_search
        inference_model = test_joint_model_infer_accuracy
        if train:
            dataset = TwoPosGridSearchDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, two_pos_weight_addr = twoPos_weight_addr, only_zero_init = True)
    elif model_name == 'TwoPosGridSearchAll':
        model = GridSearchNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/TwoPosGridSearchAll1'
        data_file_name = '2pgs_all_dataset1.npy'
        twoPos_weight_addr = '/media/okemo/extraHDD31/samueljin/Model/MLP2PosAll_best_model.pth'
        test_model = test_two_pos_grid_search
        inference_model = test_joint_model_infer_accuracy
        if train:
            dataset = TwoPosGridSearchDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, two_pos_weight_addr = twoPos_weight_addr, only_zero_init = zero_init)
    elif model_name == 'GSMSEnet':
        model = GridSearchNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/GSMSEnet1'
        data_file_name = 'grid_search_gt_mse_dataset1.npy'
        test_model = test_grid_with_MSEGT_search
        inference_model = test_aactMSE_model_infer_accuracy
        if train:
            dataset = GridSearchGTMSEDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, only_zero_init = zero_init)
    elif model_name == 'TwoPosGridSearchSparse':
        model = GridSearchNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/TwoPosGridSearchSparse1'
        data_file_name = '2pgs_sparse_dataset1.npy'
        if zero_init:
            twoPos_weight_addr = '/media/okemo/extraHDD31/samueljin/Model/MLP2Pos_best_model.pth'
        else:
            twoPos_weight_addr = '/media/okemo/extraHDD31/samueljin/Model/MLP2PosAll_best_model.pth'
        test_model = test_two_pos_grid_search
        inference_model = test_joint_model_infer_accuracy
        if train:
            dataset = TwoPosGridSearchSparseDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, two_pos_weight_addr = twoPos_weight_addr, only_zero_init = zero_init)
    else:
        print('Model not found')
        exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train:
        train_test_split = int(len(dataset) * 0.8)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_test_split, len(dataset) - train_test_split])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        train_activenet(model, train_loader, test_loader, optimizer, criterion, device, epochs, save_path)
    if test:
        if twoPos_weight_addr is None:
            test_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', folder, device)
        else:
            test_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', twoPos_weight_addr, folder, device)
    if infer:
        if twoPos_weight_addr is None:
            inference_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', folder, device)
        else:
            inference_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', twoPos_weight_addr, folder, device)
        

    