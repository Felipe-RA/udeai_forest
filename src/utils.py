import os
import json
import joblib
import numpy as np
from pprint import pprint


def convert_numpy_arrays_to_lists(obj):
    """
    Recursively convert numpy arrays to lists within a dictionary.
    
    Parameters:
        obj (dict): The dictionary to convert.
        
    Returns:
        dict: The converted dictionary.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_arrays_to_lists(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def convert_types(obj):
    """
    Recursively convert numpy arrays and numpy number types to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(e) for e in obj]
    else:
        return obj

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
    """
    Generate a report and save it as a JSON file.

    Parameters:
        report_dict (dict): The dictionary containing the report data.
        folder_path (str): The folder where the JSON file will be saved.
        base_name (str): The base name for the JSON file.
        counter (int): The counter to differentiate between different runs.

    Returns:
        str: The name of the saved JSON file.
    """
    report_file_name = f"{folder_path}/{base_name}_report{counter}.txt"
    converted_report_dict = convert_types(report_dict)
    
    with open(report_file_name, 'w') as f:
        json.dump(converted_report_dict, f, indent=4)
        
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

def save_hyperparameter_optimization_log(model, folder_path, base_name, counter):
    """
    Save the hyperparameter optimization log to a JSON file.
    
    Parameters:
        model (object): The trained model object, must have a `cv_results_` attribute.
        folder_path (str): The folder where the JSON file will be saved.
        base_name (str): The base name for the JSON file.
        counter (int): The counter to differentiate between different runs.
        
    Returns:
        str: The name of the saved JSON file.
    """
    log_file_name = f"{folder_path}/{base_name}_hyperparameter_optimization_log{counter}.json"
    
    # Convert numpy arrays to lists
    cv_results_converted = convert_numpy_arrays_to_lists(model.cv_results_)
    
    with open(log_file_name, 'w') as f:
        json.dump(cv_results_converted, f, indent=4)
        
    return log_file_name

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
    pprint(report_dict)
    print("\n\n####################\n\n")
    print(f"\nReport saved at model folder.\nFilename: {report_filename}.")
    
    # Save the model
    model_filename = save_model(model, model_folder[1], model_type, folder_number_counter)
    print(f"\nSerialized model saved at model folder.\n Filename: {model_filename}.")
    
    # Save the hyperparameters
    hyperparameters_filename = save_hyperparameters(hyperparameters_dict, model_folder[1], model_type, folder_number_counter)
    print(f"\nHyperparameters saved at model folder.\n Filename: {hyperparameters_filename}.")

    hyperparameter_optimization_log_filename = save_hyperparameter_optimization_log(model, model_folder[1], model_type, folder_number_counter)
    
    print(f"\nHyperparameter optimization log saved at model folder.\n Filename: {hyperparameter_optimization_log_filename}.")