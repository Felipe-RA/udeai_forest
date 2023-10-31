import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from copy import deepcopy
from sklearn.model_selection import KFold


from ..classes.VGGUdea import VGGUdea
from ..classes.VGGUdeaWithAdvancedTechniques import VGGUdeaWithAdvancedTechniques

# Generate some random data for demonstration purposes
# Assuming we have 1000 samples, each of shape (21, 100, 100)
# The 21 represents the number of bands
# The target variable is also randomly generated and lies between 0 and 100
# Assuming it's a regression problem
X = torch.rand((1000, 21, 100, 100))
y = torch.randint(0, 101, (1000, 1), dtype=torch.float32)

# Create a DataLoader for our training set
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define data augmentation transform for training data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # You can add more augmentation techniques here
])

# Initialize model, optimizer, and loss function
model = VGGUdeaWithAdvancedTechniques(num_bands=21)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
loss_fn = nn.L1Loss()  # MAE Loss

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Early stopping parameters
patience = 10
best_val_loss = float('inf')
counter = 0
best_model = None

# Training loop with Early Stopping, Data Augmentation, and Learning Rate Annealing
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # Apply data augmentation
        batch_X = transform(batch_X)
        
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        pred_y = model(batch_X)
        
        # Add regularization loss
        reg_loss = model.calculate_regularization_loss()
        
        loss = loss_fn(pred_y, batch_y) + reg_loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    # Validation loss for early stopping
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred_y = model(batch_X)
            loss = loss_fn(pred_y, batch_y)
            val_loss += loss.item()
    
    # Learning Rate Annealing
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            model.load_state_dict(best_model)
            break

print("Training complete.")

