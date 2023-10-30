import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from copy import deepcopy
from ..classes.VGGUdea import VGGUdea
from ..classes.VGGUdeaWithAdvancedTechniques import VGGUdeaWithAdvancedTechniques

# Generate random data for demonstration
X = torch.rand((1000, 21, 100, 100))
y = torch.randint(0, 101, (1000, 1), dtype=torch.float32)

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}")
    
    train_X = X[train_index]
    val_X = X[val_index]
    train_y = y[train_index]
    val_y = y[val_index]
    
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = VGGUdeaWithAdvancedTechniques(num_bands=21)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    loss_fn = nn.L1Loss()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize variables for early stopping
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred_y = model(batch_X)
            loss = loss_fn(pred_y, batch_y)
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
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
    print(f"Best validation loss for fold {fold + 1}: {best_val_loss}")

print("Training complete.")