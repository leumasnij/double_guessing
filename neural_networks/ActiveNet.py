import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import numpy as np
from pyro.infer import Predictive
from PyroNet import BNN_pretrained, BNN_pretrained2Pos
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------- Models ----------------------

class RegNet(nn.Module):
    def __init__(self, input_size=512, output_size=2):
        super(RegNet, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 256)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(256, 128) # hidden layer (12) -> hidden layer (24)
        self.fc3 = nn.Linear(128, 64) # hidden layer (24) -> output layer (2)
        self.fc4 = nn.Linear(64, output_size)

        
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class DenseAFNet(nn.Module):
    def __init__(self):
        super(DenseAFNet, self).__init__()
        self.fc1 = nn.Linear(8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 1)
        
        
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x 
# ---------------------- Helpers ----------------------

def normalize(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score)

def radius2degree(radius):
    return radius * 180 / np.pi

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def truedistributionfromobs(mean1, mean2, std1, std2):
    var1 = std1.copy()
    var2 = std2.copy()
    # print(mean1, mean2, std1, std2)
    for i in range(3):
        var1[i] = std1[i]**2
        var2[i] = std2[i]**2
    weighted_mean = (mean1/var1 + mean2/var2) / (1/var1 + 1/var2)
    weighted_var = 1 / (1/var1 + 1/var2)
    weighted_std = np.sqrt(weighted_var)
    return weighted_mean, weighted_std
        

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
    

def dataset_helper(folder, model_addr, deterministic_model, model = BNN_pretrained):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(model_addr, map_location=torch.device('cuda'))
    determinstic_model = deterministic_model.to(device)
    dir_list = os.listdir(folder)
    inputs = []
    targets = []
    if model == BNN_pretrained2Pos:
        for file in dir_list:
            file_path = os.path.join(folder, file)
            data_dict = np.load(file_path, allow_pickle=True, encoding= 'latin1').item()
            # zero_pad = np.zeros(8)
            # print(data_dict['force'])
            inputs.append(np.concatenate([data_dict['force'], data_dict['force']]))
            targets.append(data_dict['GT'][:3]*100)
            # print(data_dict['GT'])
        inputs = torch.tensor(inputs).float()
        inputs = inputs.view(-1, 16)
    else:
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
    print(inputs.shape)
    mean = deterministic_model(inputs).cpu().detach().numpy()
    std = outputs['obs'].std(0).cpu().detach().numpy()
    # print(std.shape)
    
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
    angle2_list = np.unique(list_of_arrays[:, -2])
    
    # closest_angle2 = -1
    # for angle2 in range(len(angle2_list)):
    #     distance = np.linalg.norm(source[-2] - angle2)
    #     if distance < smallest_distance:
    #         smallest_distance = distance
    #         closest_angle2 = angle2
            
    smallest_distance = float('inf')
    # Iterate over the list of arrays
    for i, array in enumerate(list_of_arrays):
        # Compute the average of the source and current array
        # if array[-2] == angle2_list[closest_angle2]:
            
            distance = np.linalg.norm(source - array)
            if distance < smallest_distance:
                smallest_distance = distance
                closest_index = i
    
    return closest_index

# ---------------------- Datasets ----------------------
            
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
                        input_array = [mean[indices], std[indices], inputs[i][-2:]]
                        # print(input_array)
                        input_array = np.concatenate(input_array)
                        new_est, new_std = truedistributionfromobs(mean[indices], mean[i], std[indices], std[i])
                        
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
                        new_est, new_std = truedistributionfromobs(mean[i], mean[j], std[i], std[j])
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array1, 'GT': L2_error}
                        dict2_ = {'input': input_array2, 'GT': L2_error}
                        self.data.append(dict1_)
                        self.data.append(dict2_)
            
            np.save(os.path.join(folder, data_file_name), self.data)           
        
        print(f'total data: {len(self.data)}')
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        self.data[idx]['input'][-2:] = radius2degree(self.data[idx]['input'][-2:])
        self.data[idx]['GT'] = np.linalg.norm(self.data[idx]['GT'])
        return self.data[idx]['input'], self.data[idx]['GT']

class GridSearch2Dataset(Dataset):
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None, only_zero_init = False):
        super(GridSearch2Dataset, self).__init__()
        if only_zero_init:
            data_file_name = data_file_name.split('.')[0] + '_zero_init.npy'
        if pre_orgainized:
            file_name = os.path.join(folder, data_file_name)
            self.data = np.load(file_name, allow_pickle=True)
        else:
            self.uncertainty_weight = torch.load(uncertainty_weight_addr, map_location=torch.device('cuda'))
            self.deter_model = RegNet(input_size=16, output_size=3)
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
                mean, std, inputs, targets = dataset_helper(os.path.join(folder, run), uncertainty_weight_addr, self.deter_model, BNN_pretrained2Pos)
                print(run)
                best_error = 1000
                if only_zero_init:
                    condition = (inputs[:, 6] == 0) & (inputs[:, 7] == 0)
                    indices = np.where(condition)[0]
                    # print(inputs[:, 5])
                    if len(indices) != 1:
                        
                        continue
                    for i in range(mean.shape[0]):
                        input_array = [mean[indices][0], std[indices][0], radius2degree(inputs[i][6:8])]
                        # print(input_array)
                        input_array = np.concatenate(input_array)
                        
                        x = [inputs[indices][0][:8], inputs[i][:8]]
                        x = np.concatenate(x)
                        x = torch.tensor(x).float().to(device)
                        new_est = self.deter_model(x).cpu().detach().numpy()
                        L2_error = np.linalg.norm(targets[i] - new_est)
                        dict1_ = {'input': input_array, 'GT': L2_error}
                        self.data.append(dict1_)
                        if L2_error < best_error:
                            best_error = L2_error
                    # print(f'Best error: {best_error}')
                    continue
                for i in range(mean.shape[0]):
                    for j in range(i+1, mean.shape[0]):
                        input_array1 = [mean[i], std[i], inputs[j][6:8]]
                        input_array1 = np.concatenate(input_array1)
                        input_array2 = [mean[j], std[j], inputs[i][6:8]]
                        input_array2 = np.concatenate(input_array2)
                        new_est1 = self.deter_model(torch.tensor(np.concatenate([inputs[i][:8], inputs[j][:8]])).float().to(device)).cpu().detach().numpy()
                        new_est2 = self.deter_model(torch.tensor(np.concatenate([inputs[j][:8], inputs[i][:8]])).float().to(device)).cpu().detach().numpy()
                        L2_error1 = np.linalg.norm(targets[i] - new_est1)
                        L2_error2 = np.linalg.norm(targets[j] - new_est2)
                        dict1_ = {'input': input_array1, 'GT': L2_error1}
                        dict2_ = {'input': input_array2, 'GT': L2_error2}
                        self.data.append(dict1_)
                        self.data.append(dict2_)
                        if L2_error1 < best_error or L2_error2 < best_error:
                            best_error = min(L2_error1, L2_error2)
                print(f'Best error: {best_error}')
            
            np.save(os.path.join(folder, data_file_name), self.data)           
        print(f'total data: {len(self.data)}')
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        self.data[idx]['input'][-2:] = radius2degree(self.data[idx]['input'][-2:])
        # print(self.data[idx]['input'][-2:])
        
        return self.data[idx]['input'], self.data[idx]['GT']
    
    
    
    
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
                targets = targets.float().to(device)
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
        
    
def grid_search(model, device, inputs, grid_size = 10):
    # model = model.to(device)
    angle2 = np.linspace(np.pi/6, 5*np.pi/6, grid_size)
    angle1 = np.linspace(-np.pi/2, np.pi/2, grid_size)
    output_mat = np.ones((angle1.shape[0], angle2.shape[0]))*10
    # best_angle = [0,0]
    for i in range(angle1.shape[0]):
        for j in range(angle2.shape[0]):
            input_array = [inputs, [radius2degree(angle1[i]), radius2degree(angle2[j])]]
            # input_array = [inputs, [angle1[i], angle2[j]]]
            # print(input_array)
            input_array = np.concatenate(input_array)
            # print(input_array[-2:])
            out = model(torch.tensor(input_array).float().to(device))
            if len(out) == 3:
                out = np.linalg.norm(out.cpu().detach().numpy())
            else:
                out = out.cpu().detach().numpy()
            # print(out)
            output_mat[i, j] = out
            # print(f'Angle1: {angle1[i]}, Angle2: {angle2[j]}, Output: {out}')
    ouyput_mat = output_mat.reshape(-1)
    best_angle = np.argmin(ouyput_mat)
    
    return_angle1, return_angle2 = np.unravel_index(best_angle, output_mat.shape)
    
    # print(return_angle1, return_angle2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_pos, y_pos = np.meshgrid(np.arange(output_mat.shape[1]), np.arange(output_mat.shape[0]))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    z_height = output_mat.flatten()
    x_width = y_width = 1

    ax.bar3d(x_pos, y_pos, z_pos, x_width, y_width, z_height, color='orange',zsort='average')

    ax.set_xlabel('Angle 1')
    ax.set_ylabel('Angle 2')
    ax.set_zlabel('Estimated Score')
    # ax.set_title('3D Bar Plot from 2D Array')
    plt.savefig('3d_plot.png')
    # raise ValueError('Done')
    return angle1[return_angle1], angle2[return_angle2]
            
            
def get_precentile(model, inputs, reference, targets, selected, random, device):
    in_list = []
    for input in inputs:
        x = [reference[:8], input[:8]]        
        x = np.concatenate(x)
        x = torch.tensor(x).float().to(device)
        in_list.append(x)
    in_list = torch.stack(in_list)
    out = model(in_list)
    error = np.abs(out.cpu().detach().numpy() - targets)
    error = np.linalg.norm(error, axis = 1)
    # print(error)
    selected_error = error[selected]
    random_error = error[random]
    error = np.sort(error)
    rank = np.where(error == selected_error)[0][0]
    rand_rank = np.where(error == random_error)[0][0]
    
    percent = 1 - (rank)/(len(error)-1)
    rand_percent = 1 - (rand_rank)/(len(error)-1)
    
    return percent, rand_percent


def get_precentileGS(means, reference, selected, target):
    est_list = []
    for mean in means:
        est = (mean + reference)/2
        est_list.append(est)
    est_list = np.array(est_list)
    error = np.abs(est_list - target)
    error = np.linalg.norm(error, axis = 1)
    selected_error = error[selected]
    # random_error = error[random]
    error = np.sort(error)
    rank = np.where(error == selected_error)[0][0]
    # rand_rank = np.where(error == random_error)[0][0]
    percent = 1 - (rank)/(len(error)-1)
    # rand_percent = 1 - (rand_rank)/(len(error)-1)
    return percent
        
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
    overall_percent = 0
    GT = []
    pred1 = []
    pred2 = []
    percents = []
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
                print(f'Closest idx: {closest_idx}')
                new_est, new_std = truedistributionfromobs(mean[closest_idx], mean[i], std[closest_idx], std[i])
                error_without_active = targets[i] - mean[i]
                error_with_active = targets[i] - new_est
                percent = get_precentileGS(mean, mean[i], closest_idx, targets[i])
                print(f'Percentile: {percent}')
                print(f'Error without active: {error_without_active}')
                print(f'Error with active: {error_with_active}')
                print('-----------------------------------')
                error_before_active += np.abs(error_without_active)
                error_after_active += np.abs(error_with_active)
                overall_percent += percent
                pred1.append(mean[i])
                pred2.append(new_est)
                percents.append(percent)
                GT.append(targets[i])
                if np.linalg.norm(error_with_active) < np.linalg.norm(error_without_active):
                    improve_count += 1
    print(f'Error before active: {error_before_active/len(GT)}')
    print(f'Error after active: {error_after_active/len(GT)}')
    print(f'Overall percentile: {overall_percent/len(GT)}')
    import matplotlib.pyplot as plt
    plt.figure()
    total_error_without_active = np.linalg.norm(np.array(pred1) - np.array(GT), axis = 1)
    total_error_with_active = np.linalg.norm(np.array(pred2) - np.array(GT), axis = 1)
    percentiles = np.array(percents)
    
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
    bins = np.linspace(0, 1, 20)
    plt.hist(percentiles, bins=bins)
    plt.xlabel('Percentile')
    plt.ylabel('Count')
    plt.title('Percentile distribution')
    save_name = active_model_adr.split('/')[-1].split('.')[0] + '_error_plot.png'
    print(f'Improvement count: {improve_count}')
    print(f'Improvement rate: {improve_count/len(GT)}')
    print('save_name: ', save_name)
    plt.savefig(save_name)
    result = [error_before_active/len(GT), error_after_active/len(GT), improve_count/len(GT)]
    result = np.array(result).reshape(1,3).astype(np.float64)
    np.savetxt(save_name.split('.')[0] + '_error.txt', result, fmt='%.18e', newline=' ')
    

            
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
    parser.add_argument('--epochs', '-e', type=int, default=300)
    
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
    with_raw = False
    if model_name == 'GSNet':
        model = DenseAFNet()
        save_path = '/media/okemo/extraHDD31/samueljin/Model/GSNet1'
        data_file_name = 'grid_search_dataset1.npy'
        test_model = test_grid_search
        inference_model = test_act_model_infer_accuracy
        if train:
            dataset = GridSearchDataset(folder, data_file_name = data_file_name, pre_orgainized = pre_orgainized, uncertainty_weight_addr = uncertainty_weight_addr, only_zero_init = zero_init)
    else:
        print('Model not found')
        exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train:
        train_test_split = int(len(dataset) * 0.8)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_test_split, len(dataset) - train_test_split])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        train_activenet(model, train_loader, test_loader, optimizer, criterion, device, epochs, save_path)
    if test:
        if twoPos_weight_addr is None:
            if with_raw:
                test_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', folder, device, with_raw = True)
            else:
                test_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', folder, device)
        else:
            test_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', twoPos_weight_addr, folder, device)
    if infer:
        if twoPos_weight_addr is None:
            inference_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', folder, device)
        else:
            inference_model(model, uncertainty_weight_addr, save_path + '_best_model.pth', twoPos_weight_addr, folder, device)
        

    