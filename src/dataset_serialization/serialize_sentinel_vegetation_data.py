# Importing required libraries
import os
import json
import torch
import rasterio
import numpy as np
from tqdm import tqdm

# Function to read and process a single .tif file for 'y'
def read_and_process_tif_file_for_y(filepath):
    with rasterio.open(filepath) as src:
        # Read the bands 1 and 3, skipping the empty band 2
        band1, band3 = src.read(1), src.read(3)
        
        # Apply the transformation to scale the values between 0 and 100
        band1 = (band1 / 255.0) * 100
        band3 = (band3 / 255.0) * 100
        
        # Calculate Percent_Vegetation_Coverage and clip it to be between 0 and 100
        percent_vegetation_coverage = np.clip(band1 + band3, 0, 100)
        
        # Calculate a single Percent_Vegetation_Coverage value for the entire image (e.g., mean)
        single_value = np.mean(percent_vegetation_coverage)
        
        return single_value, src.meta


# Function to read a single .tif file and return as numpy array
def read_tif_file(filepath):
    with rasterio.open(filepath) as src:
        return np.array(src.read()), src.meta

# Function to save metadata to a JSON file
def save_metadata(metadata_dict, save_path):
    for meta in metadata_dict:
        if 'sentinel_meta' in meta and 'crs' in meta['sentinel_meta']:
            meta['sentinel_meta']['crs'] = str(meta['sentinel_meta']['crs'])
        if 'treecover_meta' in meta and 'crs' in meta['treecover_meta']:
            meta['treecover_meta']['crs'] = str(meta['treecover_meta']['crs'])
    with open(save_path, 'w') as f:
        json.dump(metadata_dict, f)


# Initialize empty lists to store images and metadata
X_images = []
y_images = []
metadata_list = []


### ---------------------------- ###

# Directory paths
sentinel_dir = 'data/sentinel2rgbmedian2020.py'  # Replace with your actual directory
treecover_dir = 'data/treecover2020.py'  # Replace with your actual directory

### ---------------------------- ###


# File names are assumed to be the same in both directories
filenames = os.listdir(sentinel_dir)

# Loop through each file and read the image and metadata
for filename in tqdm(filenames):
    # Read SENTINEL-2 image
    sentinel_path = os.path.join(sentinel_dir, filename)
    sentinel_img, sentinel_meta = read_tif_file(sentinel_path)
    
    # Read and process Tree Cover image
    treecover_path = os.path.join(treecover_dir, filename)
    percent_vegetation_coverage, treecover_meta = read_and_process_tif_file_for_y(treecover_path)
    
    # Append to lists
    X_images.append(sentinel_img)
    y_images.append(percent_vegetation_coverage)
    metadata_list.append({
        'filename': filename,
        'sentinel_meta': sentinel_meta,
        'treecover_meta': treecover_meta
    })

# Convert lists to PyTorch tensors
X_tensor = torch.tensor(np.stack(X_images, axis=0))
y_tensor = torch.tensor(np.stack(y_images, axis=0), dtype=torch.float32)

# Serialize tensors and save to disk
torch.save(X_tensor, 'X_tensor.pth')
torch.save(y_tensor, 'y_tensor.pth')

# Save metadata to JSON file
save_metadata(metadata_list, 'metadata.json')

print("Data preparation and serialization complete.")
