import torch
import vbll
import torch.nn as nn
import tqdm

class hapVBLLnet(nn.Module):
    def __init__(self, input_size=6, output_size=4, dataset_size=900):
        super(hapVBLLnet, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 256)  # input layer (6) -> hidden layer (12)
        self.fc2 = nn.Linear(256, 64) # hidden layer (12) -> hidden layer (24)
        # self.fc3 = nn.Linear(64, output_size) # hidden layer (24) -> hidden layer (12)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = vbll.Regression(64, 3, 1./dataset_size)

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
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
                hap = hap.to(device)
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
                hap = hap.to(device)
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