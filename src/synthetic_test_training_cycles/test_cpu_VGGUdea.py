# First, let's simulate some training data based on the information given about MODIS and Sentinel-2 bands.
# Sentinel-2: 21 bands (for the sake of this example), 100x100 pixels
# MODIS: 2 bands (Percent_Tree_Cover, Percent_NonTree_Vegetation)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from classes.VGGUdea import VGGUdea
from classes.VGGUdeaWithAdvancedTechniques import VGGUdeaWithAdvancedTechniques

# Simulating Sentinel-2 band data: 64 samples, 21 bands, 100x100 pixels
sentinel_data = torch.rand((64, 21, 100, 100)) * 0.0001  # The values are in the range [0, 0.0001]

# Simulating MODIS data: Percent_Tree_Cover and Percent_NonTree_Vegetation
# Our target variable is Percent_Vegetation_Coverage
tree_cover = torch.rand((64,)) * 100  # Values in range [0, 100]
non_tree_vegetation = torch.rand((64,)) * 100  # Values in range [0, 100]

# Calculate the target variable
percent_vegetation_coverage = tree_cover + non_tree_vegetation
# Truncating values greater than 100
percent_vegetation_coverage[percent_vegetation_coverage > 100] = 100

# Create DataLoader
train_data = TensorDataset(sentinel_data, percent_vegetation_coverage)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Initialize our model and optimizer
model = VGGUdea(num_bands=21)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Using Mean Absolute Error Loss
criterion = nn.L1Loss()

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(100):  # Running for 50 epochs for demonstration
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1), target)
        loss.backward()
        optimizer.step()
    
    # Step the scheduler
    scheduler.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete.")