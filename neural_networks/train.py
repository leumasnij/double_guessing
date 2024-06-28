import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_helpers import Net, HapticDataset, GelResNet, GelSightDataset
import tqdm
root_dir = '/media/okemo/extraHDD31/samueljin/CoM_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Create dataset and dataloader
# dataset = HapticDataset(root_dir)
dataset = GelSightDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
model = GelResNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

lowest_loss = 1000
# Example of using the dataloader in a training loop
for epoch in range(100):  # Number of epochs
    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:  # Print loss every 10 batches
            print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Save the model
    torch.save(model.state_dict(), 'model.pth')
    if loss.item() < lowest_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        lowest_loss = loss.item()
