import os
import random
import rasterio
import matplotlib.pyplot as plt
import numpy as np

def list_tif_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.tif')]

def plot_histogram(band_data, title):
    plt.figure()
    plt.hist(band_data.flatten(), bins=255, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def explore_and_visualize_random_tif(folder_path):
    tif_files = list_tif_files(folder_path)
    if not tif_files:
        print("No .tif files found in the specified folder.")
        return

    random_tif_file = random.choice(tif_files)
    file_path = os.path.join(folder_path, random_tif_file)
    
    print(f"Picking a random .tif file: {random_tif_file}")

    with rasterio.open(file_path) as src:
        bands = []
        for i in range(1, src.count + 1):
            band_data = src.read(i)
            
            # Visualize the band
            plt.figure()
            plt.imshow(band_data, cmap='gray')
            plt.title(f"Band {i}")
            plt.colorbar()
            plt.show()

            # Print min and max values
            print("Metadata:", src.meta)
            print("Width:", src.width)
            print("Height:", src.height)
            print("CRS:", src.crs)
            print(f"Band {i} Min Value: {np.min(band_data)}, Max Value: {np.max(band_data)}")
            print(f"Band shape {band_data.shape} // Band size: {len(band_data)}")
            
            # Plot histogram
            plot_histogram(band_data, f"Histogram for Band {i}")
            
            bands.append(band_data)

        # Create a composite image if the raster has at least 2 bands
        if len(bands) >= 2:
            composite = np.dstack((bands[0], bands[1], np.zeros_like(bands[0])))
            plt.figure()
            plt.imshow(composite)
            plt.title("Composite Image")
            plt.show()

# Specify the folder path and run the function
folder_path = '../../data/treecover2020.py'
explore_and_visualize_random_tif(folder_path)
