import os
import json
import joblib

def generate_unique_folder(base_name):
    counter = 0
    while True:
        folder_name = f"{base_name}{counter}"
        folder_path = f"src/trained_models/{folder_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            return folder_name, folder_path
        counter += 1

def generate_report(report_dict, folder_path, base_name, counter):
    report_file_name = f"{folder_path}/{base_name}_report{counter}.txt"
    with open(report_file_name, 'w') as f:
        json.dump(report_dict, f)
    return report_file_name

def save_model(model, folder_path, base_name, counter):
    model_file_name = f"{folder_path}/{base_name}_model{counter}.joblib"
    joblib.dump(model, model_file_name)
    return model_file_name

def save_hyperparameters(hyperparameters_dict, folder_path, base_name, counter):
    hyperparameters_file_name = f"{folder_path}/{base_name}_hyperparameters{counter}.json"
    with open(hyperparameters_file_name, 'w') as f:
        json.dump(hyperparameters_dict, f)
    return hyperparameters_file_name

# Creating a utility function to generate the footer and save reports, models, and hyperparameters.
def save_and_report_model_artifacts(report_dict, model, hyperparameters_dict, model_folder, model_type):
    """
    Generates the footer for the training script, saves report, model, and hyperparameters.

    Parameters:
    - report_dict (dict): Dictionary containing the report information.
    - model (sklearn.BaseEstimator): Trained model.
    - hyperparameters_dict (dict): Dictionary containing the hyperparameters.
    - model_folder (str): The path where the model and reports will be saved.
    - model_type (str): The type of the model being trained.

    Returns:
    - None
    """
    folder_number_counter = model_folder[0][-1] 

    # Save the report
    report_filename = generate_report(report_dict, model_folder[1], model_type, folder_number_counter)
    
    print("\n\n####################\n\n")
    print("Report of training results: \n\n")
    print(report_dict)
    print("\n\n####################\n\n")
    print(f"\nReport saved at model folder.\nFilename: {report_filename}.")
    
    # Save the model
    model_filename = save_model(model, model_folder[1], model_type, folder_number_counter)
    print(f"\nSerialized model saved at model folder.\n Filename: {model_filename}.")
    
    # Save the hyperparameters
    hyperparameters_filename = save_hyperparameters(hyperparameters_dict, model_folder[1], model_type, folder_number_counter)
    print(f"\nHyperparameters saved at model folder.\n Filename: {hyperparameters_filename}.")
