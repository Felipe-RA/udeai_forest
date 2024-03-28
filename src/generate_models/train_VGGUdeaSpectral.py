import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit
from skorch.callbacks import EpochScoring
import numpy as np
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error
import time

# Custom utility functions
from ..utils import generate_unique_folder, save_and_report_model_artifacts

# Import the custom VGGUdeaSpectral class
from ..classes.VGGUdeaSpectral import VGGUdeaSpectral

def rmse_score(net, X, y):
    y_pred = net.predict(X)
    rmse = (mean_squared_error(y_true=y, y_pred=y_pred)) ** 0.5
    return -rmse  # Skorch tries to maximize the score, so negate the RMSE

def main():
    # Load data
    X = torch.load('X_tensor.pth')
    y = torch.load('y_tensor.pth').view(-1,1)

    # Convert X to float32
    X = X.to(dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize NeuralNetRegressor
    net = NeuralNetRegressor(
        VGGUdeaSpectral,
        module__num_bands=3,
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
        max_epochs=10,
        lr=0.01,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        train_split=ValidSplit(cv=0.2, stratified=False),
        callbacks=[EpochScoring(rmse_score, name='valid_rmse', lower_is_better=False)]
    )

    param_dist = {
        'module__num_filters1': [32, 64, 128],
        'module__num_filters2': [64, 128, 256],
        'module__num_filters3': [128, 256, 512],
        'module__activation_type': ['relu', 'sigmoid', 'tanh'],
        'module__dropout_rate': uniform(0.2, 0.5),
        'module__fc1_out_features': [512, 1024, 2048],
        'module__fc2_out_features': [256, 512, 1024],
        'lr': [0.01, 0.001, 0.0001],
        'max_epochs': [5, 10, 20]
    }

    model = RandomizedSearchCV(net, param_distributions=param_dist, n_iter=50, cv=5, verbose=3, random_state=None, scoring='neg_mean_absolute_error')

    # Train the model
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()

    # Create the report dictionary
    report_dict = {
        "Validation MAE Loss (Best)": -model.best_score_,
        "Validation RMSE (Best)": -rmse_score(model, X, y),
        "Total Training Time (seconds)": end_time - start_time,
        "Best Hyperparameters": model.best_params_,
    }

    # Create a folder for this specific model training
    model_type = "VGGUdeaSpectral"
    model_folder = generate_unique_folder(model_type)

    # Save model and report
    save_and_report_model_artifacts(report_dict, model, model.best_params_, model_folder, model_type)

if __name__ == "__main__":
    main()
