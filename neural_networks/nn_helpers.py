import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

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
    

class HapticDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_files = []
        dirlist = os.listdir(root_dir)
        for run in dirlist:
            run_dir = os.path.join(root_dir, run)
            files = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.npy')]
            self.data_files.extend(files)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data = np.load(file_path, allow_pickle=True)
        y = torch.tensor(data[0], dtype=torch.float32)
        x = torch.tensor(data[1], dtype=torch.float32)
        return x, y