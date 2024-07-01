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
    def __init__(self, input_size=512, output_size=2):
        super(RegNet, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 256)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(256, 64) # hidden layer (12) -> hidden layer (24)
        self.fc3 = nn.Linear(64, output_size) # hidden layer (24) -> hidden layer (12)

        
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
        self.model.fc = RegNet(input_size=512, output_size=2)

        
    def forward(self, x):
        # Define the forward pass
        x = self.model(x)
        return x
    

class GelRefResNet(nn.Module):
    def __init__(self):
        super(GelRefResNet, self).__init__()
        # Define the layers
        self.model = torchvision.models.resnet18(pretrained=True)
        self.fc = RegNet(input_size=512, output_size=2)
        self.model.conv1 = nn.Conv2d(
                                    in_channels=6,
                                    out_channels=self.model.conv1.out_channels,
                                    kernel_size=self.model.conv1.kernel_size,
                                    stride=self.model.conv1.stride,
                                    padding=self.model.conv1.padding,
                                    bias=False
                                )
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        
    def forward(self, x):
        # Define the forward pass
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GelHapResNet(nn.Module):
    def __init__(self):
        super(GelHapResNet, self).__init__()
        # Define the layers
        self.model = torchvision.models.resnet18(pretrained=True)
        self.fc = RegNet(input_size=512+6, output_size=2)
        self.model.conv1 = nn.Conv2d(
                                    in_channels=6,
                                    out_channels=self.model.conv1.out_channels,
                                    kernel_size=self.model.conv1.kernel_size,
                                    stride=self.model.conv1.stride,
                                    padding=self.model.conv1.padding,
                                    bias=False
                                )
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        
    def forward(self, x, hap):
        # Define the forward pass
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, hap), 1)
        x = self.fc(x)
        return x


class HapticDataset(Dataset):
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
        adrr = os.path.join(adrr, 'data.npy')
        dic = np.load(adrr, allow_pickle=True, encoding= 'latin1').item()
        x = dic['force']
        y = dic['loc'][:2]*100
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        
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
        x = cv2.resize(x, (640, 480))
        x = torch.tensor(x).float()
        x = x.permute(2, 0, 1)
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        y = torch.tensor(y).float()
        
        return x, y


class GelDifDataset(Dataset):
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
        marker_adrr = os.path.join(adrr, 'marker.png')
        x = cv2.imread(marker_adrr) - cv2.imread(os.path.join(adrr, 'marker_ref.png'))
        x = cv2.resize(x, (640, 480))
        x = torch.tensor(x).float()
        x = x.permute(2, 0, 1)
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        y = torch.tensor(y).float()
        return x, y

class GelRefDataset(Dataset):
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
        adrr1 = os.path.join(adrr, 'marker.png')
        adrr2 = os.path.join(adrr, 'marker_ref.png')
        x1 = cv2.imread(adrr1)
        x2 = cv2.imread(adrr2)
        x2 = cv2.resize(x2, (640, 480))
        x1 = cv2.resize(x1, (640, 480))
        x = np.concatenate((x1, x2), axis=2)
        x = torch.tensor(x).float()
        x = x.permute(2, 0, 1)
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        y = torch.tensor(y).float()
        return x, y
    

class GelHapDataset(Dataset):
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
        adrr1 = os.path.join(adrr, 'marker.png')
        adrr2 = os.path.join(adrr, 'marker_ref.png')
        x1 = cv2.imread(adrr1)
        x2 = cv2.imread(adrr2)
        x2 = cv2.resize(x2, (640, 480))
        x1 = cv2.resize(x1, (640, 480))
        x1 = np.concatenate((x1, x2), axis=2)
        x1 = torch.tensor(x1).float()
        x = x1.permute(2, 0, 1)
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        dic = np.load(os.path.join(self.data[idx], 'data.npy'), allow_pickle=True, encoding= 'latin1').item()
        hap = dic['force']
        hap = torch.tensor(hap).float()
        y = torch.tensor(y).float()
        return x,hap, y
# HapticDataset('/media/okemo/extraHDD31/samueljin/haptic_data')