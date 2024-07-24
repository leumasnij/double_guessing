import os
import torch
import vbll
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vbllnet import RegNet
from torch.utils.data import DataLoader
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import numpy as np
from pyro.infer import Predictive
from PyroNet import BNN_pretrained

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
    def __init__(self, folder, data_file_name = None, pre_orgainized = False, uncertainty_weight_addr = None):
        super(activeDataset, self).__init__()
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
                print(run)
                for i in range(mean.shape[0]):
                    closeset_idx = find_closest_index(mean[i], targets[i], mean)
                    
                    input_array = np.concatenate(np.array([mean[i], std[i]]))
                    # if inputs[i][-2] == 0 and inputs[i][-1] == 0:
                        # print('00_best angle: ' + str(inputs[closeset_idx][-2:]))
                    GT_ = inputs[closeset_idx][-2:]
                    dict_ = {'input': input_array, 'GT': GT_}
                    self.data.append(dict_)
            np.save(os.path.join(folder, data_file_name), self.data)
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
        
        
def test_active(uncertainty_model_adr, active_model_adr, data_folder, device):
    uncertainty_weight = torch.load(uncertainty_model_adr)
    active_model = activeNet()
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
    plt.scatter(np.array(GT)[:,1], np.array(pred2)[:,1], label = 'With Active')
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
    print(f'Improvement rate: {improve_count/GT.shape[0]}')
    print('save_name: ', save_name)
    plt.savefig(save_name)

            
    
if __name__ == '__main__':
    folder = '/media/okemo/extraHDD31/samueljin/data2'
    uncertainty_weight_addr = '/media/okemo/extraHDD31/samueljin/Model/bnn1_best_model.pth'
    # dataset = activeDataset(folder, data_file_name = 'active_dataset.npy', pre_orgainized = True, uncertainty_weight_addr = uncertainty_weight_addr)
    # model = activeNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_test_split = int(len(dataset) * 0.8)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_test_split, len(dataset) - train_test_split])
    # train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=48, shuffle=True, num_workers=4)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=0.0005)
    # criterion = torch.nn.MSELoss()
    save_path = '/media/okemo/extraHDD31/samueljin/Model/activeNet'
    # train_activenet(model, train_loader, test_loader, optimizer, criterion, device, 500, save_path)
    test_active(uncertainty_weight_addr, save_path + '_best_model.pth', folder, device)
    