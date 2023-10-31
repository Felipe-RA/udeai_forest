import os
import rasterio
from collections import defaultdict
from tqdm import tqdm

def list_tif_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.tif')]

def explore_folder(folder_path):
    config_count = defaultdict(int)
    
    tif_files = list_tif_files(folder_path)
    for tif_file in tqdm(tif_files, desc=f"Processing {folder_path}"): 
        file_path = os.path.join(folder_path, tif_file)
        with rasterio.open(file_path) as src:
            config = (
                src.width,
                src.height,
                tuple(src.dtypes),
                str(src.crs),
                src.count
            )
            config_count[config] += 1
    
    return config_count

def explore_tif_folders(folder_paths):
    for folder_path in folder_paths:
        absolute_folder_path = os.path.abspath(folder_path)
        print(f"\n\n\nExploring folder: {absolute_folder_path}")

        config_count = explore_folder(folder_path)
        print("=======================================")
        print("\nUnique Configurations:")
        for i, (config, count) in enumerate(config_count.items(), 1):
            width, height, dtypes, crs, bands = config
            print(f"\nConfiguration #{i}:")
            print(f"  Width: {width}, Height: {height}")
            print(f"  Data Types: {dtypes}")
            print(f"  CRS: {crs}")
            print(f"  Number of Bands: {bands}")
            print(f"  Count of Images with this Configuration: {count}")
        print("=======================================\n")


# Initial folder paths, relative to the current script location

tif_dataset_paths = [
    'data/sentinel2rgbmedian2020.py',
    'data/treecover2020.py'
]
explore_tif_folders(tif_dataset_paths)
