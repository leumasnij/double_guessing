import torch
import vbll
import torch.nn as nn
from tqdm import tqdm
from nn_helpers import HapticDataset, HapDatasetFromTwoPos, HapOnePos, HapTwoPos
from torch.utils.data import DataLoader
import numpy as np
import os
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
    
    
class hapVBLLnet(nn.Module):
    def __init__(self, input_size=8, output_size=3, dataset_size=9000):
        super(hapVBLLnet, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 256)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(256, 128) # hidden layer (12) -> hidden layer (24)
        # self.fc3 = nn.Linear(64, output_size) # hidden layer (24) -> hidden layer (12)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = vbll.Regression(64, output_size, 5./dataset_size, prior_scale= .1, wishart_scale=1e-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    
    
def train_vbll(model, train_loader, val_loader, optimizer, device, epochs, save_path):
    lowest_loss = 1000
    for epoch in range(epochs):  # Number of epochs
        cum_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                optimizer.zero_grad()
                inputs = inputs.to(device)
                # hap = hap.to(device)
                targets = targets.to(device)
                outputs = model(inputs)  # For GelHapResNet
                # Compute the loss
                loss = outputs.train_loss_fn(targets)
                cum_loss += loss.item()
                # Backward pass
                # optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                pbar.set_postfix({'loss': cum_loss/(i+1)})
                pbar.update(1)
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                # hap = hap.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = outputs.train_loss_fn(targets)
                val_loss += loss.item()
        print(f"Validation loss: {val_loss/len(val_loader)}")
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(model.state_dict(), save_path+'_best_model.pth')
        torch.save(model.state_dict(), save_path+'_model.pth')
    print('Finished Training')
    error = np.array([0.,0.,0.])
    x_error = np.array([])
    y_error = np.array([])
    z_error = np.array([])
    x_std = np.array([])
    y_std = np.array([])
    z_std = np.array([])
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            # targets = targets.to(device)
            # hap = hap.to(device)
            predictions = model(inputs).predictive
            mean = predictions.mean.cpu().numpy()
            cov = predictions.covariance
            print(mean.shape, cov.shape)
            targets = targets.cpu().numpy()
            batch_error = np.abs(mean[:, 0:3] - targets[:, 0:3])
            std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1)).cpu().numpy()
            error += batch_error.sum(axis=0)
            x_error = np.append(x_error, batch_error[:,0])
            y_error = np.append(y_error, batch_error[:,1])
            z_error = np.append(z_error, batch_error[:,2])
            x_std = np.append(x_std, std[:,0])  
            y_std = np.append(y_std, std[:,1])
            z_std = np.append(z_std, std[:,2])
            
    # error = np.array(error)
    print(f"Average error: {error/(len(val_loader)*48)}")
    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(1,3,1)
    plt.scatter(x_error, x_std)
    plt.subplot(1,3,2)
    plt.scatter(y_error, y_std)
    plt.subplot(1,3,3)
    plt.scatter(z_error, z_std)
    plt.savefig('error_vs_std.png')
    
    
def trainHap(model, train_loader, val_loader, optimizer, device, epochs, save_path):
    lowest_loss = 1000
    for epoch in range(epochs):  # Number of epochs
        cum_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                optimizer.zero_grad()
                inputs = inputs.to(device)
                # hap = hap.to(device)
                targets = targets.to(device)
                outputs = model(inputs)  # For GelHapResNet
                # Compute the loss
                loss = torch.nn.functional.mse_loss(outputs, targets)
                cum_loss += loss.item()
                # Backward pass
                # optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                pbar.set_postfix({'loss': cum_loss/(i+1)})
                pbar.update(1)
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                # hap = hap.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                val_loss += loss.item()
        print(f"Validation loss: {val_loss/len(val_loader)}")
        val_loss /= len(val_loader)
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(model.state_dict(), save_path+'_best_model.pth')
        torch.save(model.state_dict(), save_path+'_model.pth')
    print('Finished Training')
    
    test_model(model, val_loader, device)
    
def test_model(model, val_loader, device):
    error = [0,0,0]
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            # hap = hap.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_error = torch.abs(outputs - targets).sum(dim=0).cpu().numpy()
            error = np.add(error, batch_error)
    batch_size = 48
    print(f"Average error: {error/(len(val_loader)*batch_size)}")
    
def save_and_eval(model, folder, device, save_path):
    model_input = model.fc1.in_features
    save_dict = {}
    overall_error = [0,0,0]
    average_best_error = [0,0,0]
    for file1 in os.listdir(folder):
        save_array = []
        if '.' in file1:
           continue
        if len(os.listdir(os.path.join(root_dir, file1))) == 0:
            continue
        dataset = []
        GT = []
        for file2 in os.listdir(os.path.join(root_dir, file1)):
            dict_ = np.load(os.path.join(root_dir, file1, file2), allow_pickle=True).item()
            dataset.append(dict_['force'])
            GT.append(dict_['GT'][:3]*100)
        dataset = np.array(dataset)
        condition = (dataset[:, -2] == 0) & (dataset[:, -1] == 0)
        indices = np.where(condition)[0]
        if len(indices) != 1:
            continue
        reference = dataset[indices[0]]
        dataset = np.delete(dataset, indices, axis=0)
        GT = np.array(GT)
        GT = np.delete(GT, indices, axis=0)
        
        # print(reference)
        if model_input == 16:
        #put reference every item in the dataset at the beginning
            new_dataset = np.zeros((dataset.shape[0], 16))
            for i in range(dataset.shape[0]):
                new_dataset[i] = np.concatenate((reference, dataset[i]))
        else:
            new_dataset = np.zeros((dataset.shape[0], 14))
            for i in range(dataset.shape[0]):
                new_dataset[i] = np.concatenate((reference[:6], dataset[i]))
        # print(new_dataset[0])
        new_dataset = torch.tensor(new_dataset).float()
        new_dataset = new_dataset.to(device)
        GT = torch.tensor(GT).float().to(device)
        with torch.no_grad():
            outputs = model(new_dataset)
            for i in range(len(outputs)):
                error = torch.abs(outputs[i] - GT[i]).cpu().numpy()
                error = np.linalg.norm(error)
                save_array.append([new_dataset[i][-2:].cpu().numpy(), error])
            average_error = torch.abs(outputs - GT).mean(dim=0).cpu().numpy()
            best_error = torch.abs(outputs - GT).min(dim=0).values.cpu().numpy()
            print(f"Average error for {file1}: {average_error}")
            print(f"Best error for {file1}: {best_error}")
            overall_error = np.add(overall_error, average_error)
            average_best_error = np.add(average_error, best_error)
        save_dict[file1] = save_array
    np.save(save_path, save_dict)
    print(f"Overall error: {overall_error/len(save_dict)}")
    print(f"Average best error: {average_best_error/len(save_dict)}")
        

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = hapVBLLnet().to(device)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=0.0005)
    # root_dir = '/media/okemo/extraHDD31/samueljin/data2'
    # save_path = '/media/okemo/extraHDD31/samueljin/Model/vbllnetOnePos'
    # dataset = HapOnePos(root_dir)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=48, shuffle=True, num_workers=4)
    # train_vbll(model, train_loader, val_loader, optimizer, device, 50, save_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = RegNet(input_size=14, output_size=3).to(device)
    model = RegNet(input_size=16, output_size=3).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    root_dir = '/media/okemo/extraHDD31/samueljin/data2'
    save_path = '/media/okemo/extraHDD31/samueljin/Model/MLP2Pos'
    # dataset = HapDatasetFromTwoPos(root_dir)
    dataset = HapTwoPos(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=True, num_workers=4)
    # train_vbll(model, train_loader, val_loader, optimizer, device, 500, save_path)
    # trainHap(model, train_loader, val_loader, optimizer, device, 500, save_path)
    # model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/MLP2Pos_best_model.pth'))
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/MLP2PosAll_best_model.pth'))
    test_model(model, val_loader, device)
    save_path = '/media/okemo/extraHDD31/samueljin/data2/resultsALL.npy'
    save_and_eval(model, root_dir, device, save_path=save_path)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = RegNet(input_size=8, output_size=3).to(device)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    # root_dir = '/media/okemo/extraHDD31/samueljin/data2'
    # save_path = '/media/okemo/extraHDD31/samueljin/Model/RegNetOnePos'
    # dataset = HapOnePos(root_dir)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    # trainHap(model, train_loader, val_loader, optimizer, device, 50, save_path)