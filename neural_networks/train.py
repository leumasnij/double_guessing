import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from neural_networks.nn_helpers import Net, HapticDataset

root_dir = 'path/to/haptic_data'

# Create dataset and dataloader
dataset = HapticDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Example of using the dataloader in a training loop
for epoch in range(10):  # Number of epochs
    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:  # Print loss every 10 batches
            print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')