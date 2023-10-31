import os
import random
import rasterio
import matplotlib.pyplot as plt
import numpy as np

def list_tif_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.tif')]

def plot_histogram(band_data, title):
    plt.hist(band_data.flatten(), bins=255)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def explore_and_visualize_tif(file_path):
    with rasterio.open(file_path) as src:
        print(f"\nExploring file: {os.path.basename(file_path)}")
        print(f"Width, height: {src.width}, {src.height}")
        print(f"Data types: {tuple(src.dtypes)}")
        print(f"Coordinate Reference System: {src.crs}")
        print(f"Number of bands: {src.count}")

        # Visualize each band separately
        for i in range(1, src.count + 1):
            band_data = src.read(i)
            plt.imshow(band_data, cmap='gray')
            plt.title(f"Band {i}")
            plt.colorbar()
            plt.show()

            # Plot histogram for the band
            print("\nMetadata:", src.meta,"\n")
            print("Width:", src.width)
            print("Height:", src.height)
            print("CRS:", src.crs)
            print(f"Band shape {band_data.shape} // Band size: {len(band_data)}")
            plot_histogram(band_data, f"Histogram for Band {i}")

        # Create a composite visualization if the image has at least 3 bands
        if src.count >= 3:
            # Read the first three bands
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

            # Create an RGB image
            rgb = np.dstack((red, green, blue))
            plt.imshow(rgb)
            plt.title("RGB Composite")
            plt.show()



folder_path = 'data/sentinel2-rgb-median-2020'


tif_files = list_tif_files(folder_path)

if not tif_files:
    print("No .tif files found in the specified folder.")
else:
    print("Picking a random .tif file and exploring its content...")
    random_tif_file = random.choice(tif_files)
    file_path = os.path.join(folder_path, random_tif_file)
    explore_and_visualize_tif(file_path)
