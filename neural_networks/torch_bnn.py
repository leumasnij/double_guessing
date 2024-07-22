import torch
import vbll
import torch.nn as nn
import torchbnn as bnn
import numpy as np
from tqdm import tqdm
from nn_helpers import HapticDataset, RegNet, HapDatasetFromTwoPos, HapOnePos
from torch.utils.data import DataLoader

class BayesianMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(BayesianMLP, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_size, out_features=256)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=128)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=128, out_features=64)
        self.fc4 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=64, out_features=output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def trainBayesianMLP(model, train_loader, val_loader, optimizer, device, epochs, save_path):
    lowest_loss = 1000
    KL_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    for epoch in range(epochs):  # Number of epochs
        cum_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                optimizer.zero_grad()
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)  # For GelHapResNet
                # Compute the loss
                loss = nn.MSELoss()(outputs, targets)
                cum_loss += loss.item()
                klloss = KL_loss(model)*0.1
                
                loss = loss + klloss
                # Backward pass
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': cum_loss/(i+1)})
                pbar.update(1)
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                val_loss += loss.item()
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                torch.save(model.state_dict(), save_path+'_best_model.pth')
            torch.save(model.state_dict(), save_path+'_model.pth')
        print(f"Validation loss: {val_loss/len(val_loader)}")
    print('finished training')
    
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
            targets = targets.to(device)
            mean, std = bnn_predict(model, inputs)
            batch_error = np.abs(targets.cpu().numpy() - mean)
            # print(f"Batch error: {batch_error.sum(axis=0)}")
            error += batch_error.sum(axis=0)
            x_error = np.append(x_error, batch_error[:,0])
            y_error = np.append(y_error, batch_error[:,1])
            z_error = np.append(z_error, batch_error[:,2])
            x_std = np.append(x_std, std[:,0])  
            y_std = np.append(y_std, std[:,1])
            z_std = np.append(z_std, std[:,2])
            
    # error = np.array(error)
    print(f"Average error: {error/len(val_loader)}")
    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(1,3,1)
    plt.scatter(x_error, x_std)
    plt.subplot(1,3,2)
    plt.scatter(y_error, y_std)
    plt.subplot(1,3,3)
    plt.scatter(z_error, z_std)
    plt.savefig('torchbnn.png')
    
def bnn_predict(model, X, n_iter=500):
    model.eval()
    predictions = []

    for _ in range(n_iter):
        with torch.no_grad():
            preds = model(X).cpu().numpy()
            predictions.append(preds)

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    return mean_prediction, uncertainty   
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BayesianMLP(input_size=8, output_size=3).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0025)
    root_dir = '/media/okemo/extraHDD31/samueljin/data2'
    save_path = '/media/okemo/extraHDD31/samueljin/Model/torch_bnn'
    dataset = HapOnePos(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=True, num_workers=4)
    trainBayesianMLP(model, train_loader, val_loader, optimizer, device, 100, save_path)