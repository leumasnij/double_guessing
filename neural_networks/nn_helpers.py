import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(6, 18)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(18, 36) # hidden layer (12) -> hidden layer (24)
        self.fc3 = nn.Linear(36, 64) # hidden layer (24) -> hidden layer (12)
        self.fc4 = nn.Linear(64, 3)  # hidden layer (12) -> output layer (3)
        
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer, no activation function (for regression tasks)
        return x

class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(512, 256)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(256, 64) # hidden layer (12) -> hidden layer (24)
        self.fc3 = nn.Linear(64, 2) # hidden layer (24) -> hidden layer (12)

        
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer, no activation function (for regression tasks)
        return x


class GelResNet(nn.Module):
    def __init__(self):
        super(GelResNet, self).__init__()
        # Define the layers
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = RegNet()

        
    def forward(self, x):
        # Define the forward pass
        x = self.model(x)
        return x


class HapticDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        dirlist = os.listdir(root_dir)
        for run in dirlist:
            run_dir = os.path.join(root_dir, run)
            for file in os.listdir(run_dir):
                data_file = os.path.join(run_dir, file)
            
                data = np.load(data_file, allow_pickle=True, encoding= 'latin1').item()
                self.data.append(data)
                # print(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        y = item['GT']
        x = item['force']
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()*1000
        return x, y

class GelSightDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        dirlist = os.listdir(root_dir)
        for run in dirlist:
            run_dir = os.path.join(root_dir, run)
            self.data.append(run_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        adrr = self.data[idx]
        adrr = os.path.join(adrr, 'marker.png')
        x = cv2.imread(adrr)
        x = cv2.resize(x, (480, 640))
        x = torch.tensor(x).float()
        x = x.permute(2, 0, 1)
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        y = torch.tensor(y).float()
        return x, y


class GelDefDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        dirlist = os.listdir(root_dir)
        for run in dirlist:
            run_dir = os.path.join(root_dir, run)
            self.data.append(run_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        adrr = self.data[idx]
        adrr = os.path.join(adrr, 'marker.png')
        x = cv2.imread(adrr) - cv2.imread(os.path.join(adrr, 'marker_ref.png'))
        x = cv2.resize(x, (480, 640))
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        return x, y


# HapticDataset('/media/okemo/extraHDD31/samueljin/haptic_data')