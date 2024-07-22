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
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer, no activation function (for regression tasks)
        return x
class HapNetWithUncertainty(nn.Module):
    def __init__(self, input_size=6, output_size=4):
        super(HapNetWithUncertainty, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 256)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(256, 64) # hidden layer (12) -> hidden layer (24)
        self.fc3 = nn.Linear(64, output_size) # hidden layer (24) -> hidden layer (12)
        self.dropout = nn.Dropout(0.2)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        mean1 = x[..., 0][..., None]
        mean2 = x[..., 1][..., None]
        std1 = torch.clamp(x[..., 2][..., None], min=1e-3)
        std2 = torch.clamp(x[..., 3][..., None], min=1e-3)
        normal1 = torch.distributions.Normal(mean1, std1)
        normal2 = torch.distributions.Normal(mean2, std2)
        # std = torch.clamp(x[..., 2:][..., None], min=1e-3)
        # normal = torch.distributions.Normal(mean, std)
        return [normal1, normal2]


def HapticLoss(dist, target):

    dist1 = dist[0]
    dist2 = dist[1]
    target1 = target[..., 0][..., None]
    target2 = target[..., 1][..., None]
    return -dist1.log_prob(target1).mean() - dist2.log_prob(target2).mean()
    # return -dist.log_prob(target).sum()

class SmallerCNN(nn.Module):
    def __init__(self):
        super(SmallerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=  'same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, stride=1, padding= 'same')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        # Flattening the final output to match 512x1 feature map
        x = x.view(-1, 512*1)  # Flatten to 512
        return x



class GelResNet(nn.Module):
    def __init__(self):
        super(GelResNet, self).__init__()
        # Define the layers
        self.model = torchvision.models.resnet18(pretrained=True)
        # self.model = SmallerCNN()
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.fc = RegNet(input_size=512, output_size=2)

        
    def forward(self, x):
        # Define the forward pass
    
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
        # self.model = SmallerCNN()
        self.fc = RegNet(input_size=512+6, output_size=2)
        self.model.conv1 = nn.Conv2d(in_channels=6,out_channels=self.model.conv1.out_channels,kernel_size=self.model.conv1.kernel_size,stride=self.model.conv1.stride,padding=self.model.conv1.padding,bias=False)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        
    def forward(self, x, hap):
        # Define the forward pass
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, hap), 1)
        x = self.fc(x)
        return x

class GelHapLite(nn.Module):
    def __init__(self):
        super(GelHapLite, self).__init__()
        # Image processing layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to 1x1

        # Fully connected layers for combining features and numerical input
        self.fc1 = nn.Linear(256 + 6, 128)  # 128 from CNN, 6 from numerical input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output two regression values
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, image, nums):
        # Process image
        x =  self.pool(self.relu(self.conv1(image)))
        x =  self.pool(self.relu(self.conv2(x)))
        x =  self.pool(self.relu(self.conv3(x)))
        x =  self.pool(self.relu(self.conv4(x)))
        x = self.adapool(x)
        x = x.view(x.size(0), -1)  # Flatten to 1D
        # Combine image features with numerical data
        x = torch.cat((x, nums), dim=1)  # Concatenate features and numerical input along the feature dimension

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer for regression
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
        # adrr = os.path.join(adrr, 'data.npy')
        dic = np.load(adrr, allow_pickle=True, encoding= 'latin1').item()
        x = dic['force']
        y = dic['loc']*100
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
        x = x/255.0
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
        x = x/255.0
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
        x = x/255.0
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
        x1 = x1/255.0
        x1 = torch.tensor(x1).float()
        x = x1.permute(2, 0, 1)
        y = np.loadtxt(os.path.join(self.data[idx], 'loc.txt'))[:2]*100
        dic = np.load(os.path.join(self.data[idx], 'data.npy'), allow_pickle=True, encoding= 'latin1').item()
        hap = dic['force']
        hap = torch.tensor(hap).float()
        y = torch.tensor(y).float()
        return x,hap, y

class HapDatasetFromTwoPos(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        dirlist = os.listdir(root_dir)
        for run in dirlist:
            dataset_length = len(os.listdir(os.path.join(root_dir, run)))
            ref_file_name = 'data' + str(dataset_length-1) + '.npy'
            ref_force = np.load(os.path.join(root_dir, run, ref_file_name), allow_pickle=True, encoding= 'latin1').item()['force'][:6]
            # print(np.load(os.path.join(root_dir, run, ref_file_name), allow_pickle=True, encoding= 'latin1').item()['GT'])
            # print(ref_force)
            if sum(ref_force) == 0:
                # print('skipping ' + os.path.join(root_dir, run, ref_file_name))
                continue
            for data in os.listdir(os.path.join(root_dir, run)):
                if data == ref_file_name:
                    # print('skipping ' + os.path.join(root_dir, run, data))
                    continue
                run_dir = os.path.join(root_dir, run, data)
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
        
class HapOnePos(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        dirlist = os.listdir(root_dir)
        for run in dirlist:
            dataset_length = len(os.listdir(os.path.join(root_dir, run)))
            for data in os.listdir(os.path.join(root_dir, run)):

                run_dir = os.path.join(root_dir, run, data)
                dict_ = np.load(run_dir, allow_pickle=True, encoding= 'latin1').item()
                if sum(np.abs(dict_['force'][:6])) == 0:
                    # print('skipping ' + run_dir)
                    os.remove(run_dir)
                    
                    continue
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
        
