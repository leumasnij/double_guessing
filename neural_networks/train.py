import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_helpers import Net, HapticDataset, GelResNet, GelSightDataset, GelDifDataset, GelRefDataset, GelHapDataset, GelHapResNet, RegNet, GelRefResNet, GelHapLite, HapNetWithUncertainty, HapticLoss
from tqdm import tqdm
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model', '-m', type=str)
model_name = argparser.parse_args().model
epochs = 50
# torch.manual_seed(3407)
root_dir = '/media/okemo/extraHDD31/samueljin/CoM_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
criterion = nn.MSELoss()
if model_name == 'gel':
    dataset = GelSightDataset(root_dir)
    model = GelResNet().to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/gel'
elif model_name == 'haptic':
    dataset = HapticDataset(root_dir)
    model = RegNet(input_size=6).to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/hap'
elif model_name == 'diff':
    dataset = GelDifDataset(root_dir)
    model = GelResNet().to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/dif'
elif model_name == 'ref':
    dataset = GelRefDataset(root_dir)
    model = GelRefResNet().to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/ref'
elif model_name == 'gelhap':
    dataset = GelHapDataset(root_dir)
    model = GelHapResNet().to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/gelhap'
elif model_name == 'gelhaplite':
    dataset = GelHapDataset(root_dir)
    model = GelHapLite().to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/gelhaplite'
elif model_name == 'hap_uncertain':
    dataset = HapticDataset(root_dir)
    model = HapNetWithUncertainty(input_size=6).to(device)
    save_path = '/media/okemo/extraHDD31/samueljin/Model/hap_uncertain'
    criterion = HapticLoss
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
val_loader = DataLoader(test_dataset, batch_size=48, shuffle=True, num_workers=4)
# dataloader = DataLoader(dataset, batch_size=48, shuffle=True, num_workers=4)


optimizer = optim.SGD(model.parameters(), lr=0.02)
# optimizer = optim.RAdam(model.parameters(), lr=60e-4)

lowest_loss = 1000
# Example of using the dataloader in a training loop
if model_name == 'gelhap' or model_name == 'gelhaplite':
    cum_loss = 0
    for epoch in range(epochs):  # Number of epochs
        cum_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/50") as pbar:
            for i, (inputs, hap, targets) in enumerate(train_loader):
                # Forward pass
                inputs = inputs.to(device)
                hap = hap.to(device)
                targets = targets.to(device)
                outputs = model(inputs, hap)  # For GelHapResNet
                # Compute the loss
                loss = criterion(outputs, targets)
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        cum_loss /= len(train_loader)
        print(f"Epoch {epoch+1} loss: {cum_loss}")
        model.eval()
        cum_loss = 0
        with torch.no_grad():
            for i, (inputs, hap, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                hap = hap.to(device)
                targets = targets.to(device)
                outputs = model(inputs, hap)
                loss = criterion(outputs, targets)
                cum_loss += loss.item()
        cum_loss /= len(val_loader)
        print(f"Validation loss: {cum_loss}")
        model.train()
        # Save the model
        torch.save(model.state_dict(), save_path + '_model.pth')
        if cum_loss < lowest_loss:
            torch.save(model.state_dict(), save_path + '_best_model.pth')
            lowest_loss = loss.item()
else:
    
    for epoch in range(epochs):  # Number of epochs
        cum_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/50") as pbar:
            for i, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                # Compute the loss
                loss = criterion(outputs, targets)
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
                
                if (i+1) % 1 == 0:  # Print loss every 10 batches
                    pbar.set_postfix(loss=loss.item())
                
                pbar.update(1)
        cum_loss /= len(train_loader)
        print(f"Epoch {epoch+1} loss: {cum_loss}")
        model.eval()
        cum_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                cum_loss += loss.item()
        cum_loss /= len(val_loader)
        print(f"Validation loss: {cum_loss}")
        # Save the model
        torch.save(model.state_dict(), save_path + '_model.pth')
        if cum_loss < lowest_loss:
            torch.save(model.state_dict(), save_path + '_best_model.pth')
            lowest_loss = loss.item()