from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import os
import json
import joblib
import numpy as np
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# Importing the custom model class
from ..classes.MultipleRegressionModel import MultipleRegressionModel

# Utility functions
from ..utils import *

def main():
    # Load the data
    X = torch.load('X_tensor.pth').numpy()
    y = torch.load('y_tensor.pth').numpy()
    
    # Flatten the images
    X = X.reshape(X.shape[0], -1)
    
    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=2, shuffle=True, random_state=None)



    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False]
    }

    # Initialize the model with hyperparameter grid
    model = GridSearchCV(
                        estimator=MultipleRegressionModel(),
                        param_grid=param_grid,
                        n_jobs=2,            # really unstable. move back to 1 if crashing. dont use -1
                        scoring='neg_mean_absolute_error',  #  minimize MAE
                        cv=3,                
                        verbose=3    
                    )
    
    # Initialize variables for cross-validation
    fold_mae_losses = []
    fold_rmse_losses = []
    
    # Create a folder for this specific model training
    model_type = "MultipleRegression"
    model_folder = generate_unique_folder(str(model_type))
    print(f"Folder {model_folder[0]} created at {model_folder[1]}")
    
    # Cross-validation loop
    for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X), desc='KFold Progress')):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Train the model on the training set
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        # Make predictions on the validation set
        y_pred = model.predict(X_val)
        
        # Calculate the MAE loss
        mae_loss = mean_absolute_error(y_val, y_pred)
        fold_mae_losses.append(mae_loss)

        # Calculate the RMSE loss
        rmse_loss = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_rmse_losses.append(rmse_loss)
      
        # Report for the current fold
        print(f"\tFold {fold + 1}, Validation MAE Loss: {mae_loss}, RMSE Loss: {rmse_loss}")
        
    # Calculate the average and standard deviation of the losses over all folds
    avg_mae_loss = np.mean(fold_mae_losses)
    std_mae_loss = np.std(fold_mae_losses)
    avg_rmse_loss = np.mean(fold_rmse_losses)
    std_rmse_loss = np.std(fold_rmse_losses)

    # Create the report dictionary
    report_dict = {
        "Validation MAE Loss (Avg)": avg_mae_loss,
        "Validation MAE Loss (Std)": std_mae_loss,
        "Validation RMSE Loss (Avg)": avg_rmse_loss,
        "Validation RMSE Loss (Std)": std_rmse_loss,
        "Total Training Time (seconds)": end_time - start_time,
        "Best Hyperparameters": model.best_params_
    }
    
    

    save_and_report_model_artifacts(report_dict, model, model.best_params_, model_folder, model_type)

if __name__ == "__main__":
    main()
