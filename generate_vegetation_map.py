import os
import torch
import joblib
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from osgeo import gdal
from shapely.geometry import box
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse

# Import custom model definitions
from src.classes.VGGUdeaSpectral import VGGUdeaSpectral
from src.classes.MultipleRegressionModel import MultipleRegressionModel







def save_metadata(metadata_list, save_path):
    # Modify metadata to serialize
    for meta in metadata_list:
        if 'sentinel_meta' in meta:
            if 'crs' in meta['sentinel_meta']:
                # Convert CRS object to string representation
                meta['sentinel_meta']['crs'] = str(meta['sentinel_meta']['crs'])
    with open(save_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)

def create_file_list(directory, output_file):
    """
    Generates a list of file paths for TIFF files in the specified directory and writes it to an output file.
    Args:
        directory (str): Directory containing the TIFF files.
        output_file (str): File path where the list of TIFF file paths will be saved.
    """
    with open(output_file, 'w') as file_list:
        for filename in os.listdir(directory):
            if filename.endswith('.tif'):
                file_path = os.path.join(directory, filename)
                file_list.write(file_path + '\n')

def rmse_score(net, X, y):
    """
    Compute the root mean squared error, negated, as Skorch uses negative scores for minimization.
    
    Args:
        net: The neural network model.
        X: Input data as tensors.
        y: True labels.
        
    Returns:
        float: The negated root mean squared error.
    """
    y_pred = net.predict(X)
    return -(mean_squared_error(y_true=y, y_pred=y_pred) ** 0.5)

def save_prediction_tile(prediction, metadata, output_dir, filename):
    """
    Saves a prediction value as a TIFF tile with updated metadata based on the input metadata.
    
    Args:
        prediction (float): The predicted value to fill the tile.
        metadata (dict): The metadata from the original raster file.
        output_dir (str): Directory where the prediction TIFF will be saved.
        filename (str): Filename for the output TIFF.
    """
    prediction_tile = np.full((100, 100), prediction, dtype=np.float32)
    new_meta = metadata.copy()
    new_meta.update({
        'dtype': 'float32',
        'count': 1,
        'driver': 'GTiff'
    })
    output_path = os.path.join(output_dir, filename)
    with rasterio.open(output_path, 'w', **new_meta) as dest:
        dest.write(prediction_tile, 1)

def gdalbuildvrt(vrt_path, directory):
    """
    Generates a VRT (Virtual Raster) file from all TIFF files in a specified directory.
    
    Args:
        vrt_path (str): Path where the VRT file will be saved.
        directory (str): Directory containing the TIFF files to be included in the VRT.
    """
    file_list_path = os.path.join(directory, 'file_list.txt')
    with open(file_list_path, 'w') as file_list:
        for filename in os.listdir(directory):
            if filename.endswith('.tif'):
                file_path = os.path.join(directory, filename)
                file_list.write(file_path + '\n')
    os.system(f"gdalbuildvrt -input_file_list {file_list_path} {vrt_path}")

def convert_vrt_to_png(vrt_path, png_path, width, height):
    """
    Converts a Virtual Raster (VRT) file to a PNG image with specified dimensions.
    
    Args:
        vrt_path (str): Path to the VRT file to convert.
        png_path (str): Path where the PNG image will be saved.
        width (int): Width of the PNG image.
        height (int): Height of the PNG image.
    """
    dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
    options = gdal.TranslateOptions(format='PNG', outputType=gdal.GDT_Byte, width=width, height=height)
    gdal.Translate(png_path, dataset, options=options)
    dataset = None  # Close dataset to free resources

def generate_predictions_and_visualizations(data_dir, model_path, output_dir, enable_tiles=False, width=2560, height=1440):
    """
    Processes TIFF files to generate predictions using a pre-trained model and optionally generates
    prediction tiles, a Virtual Raster (VRT), and a PNG image of these tiles.

    Args:
        data_dir (str): Directory containing TIFF files.
        model_path (str): Path to the trained model.
        output_dir (str): Directory to save outputs.
        enable_tiles (bool): If True, generates prediction tiles and related outputs.
        width (int): Width of the output PNG image if tiles are generated.
        height (int): Height of the output PNG image if tiles are generated.

    Returns:
        dict: Contains 'data' (GeoDataFrame with IDs, geometries, and predictions),
              'vrt' (path to VRT if generated), and 'png' (path to PNG if generated).
    """
    model = joblib.load(model_path)
    os.makedirs(output_dir, exist_ok=True)
    if enable_tiles:
        prediction_tiles_dir = os.path.join(output_dir, "prediction_tiles")
        os.makedirs(prediction_tiles_dir, exist_ok=True)
        file_list_path = os.path.join(prediction_tiles_dir, 'file_list.txt')


    results_gdf = gpd.GeoDataFrame(columns=["id", "geometry", "prediction"])
    for filename in tqdm(os.listdir(data_dir), desc="Processing TIFF files"):
        if filename.endswith('.tif'):
            filepath = os.path.join(data_dir, filename)
            with rasterio.open(filepath) as src:
                img_array = src.read().astype('float32')
                image_tensor = torch.tensor(img_array).unsqueeze(0)
                prediction_tensor = model.predict(image_tensor)
                
                # Ensure prediction tensor is correctly handled based on its dimensions
                if prediction_tensor.ndim > 1 and prediction_tensor.shape[0] == 1:
                    prediction = float(prediction_tensor.squeeze().item())  # Extracting a single value
                else:
                    prediction = float(prediction_tensor.item())  # Direct extraction

                new_data = {
                    "id": filename,
                    "geometry": box(*src.bounds),
                    "prediction": prediction
                }

                new_gdf = gpd.GeoDataFrame([new_data], columns=["id", "geometry", "prediction"], geometry='geometry')
                
                if results_gdf.empty:
                    results_gdf = new_gdf
                else:
                    results_gdf = pd.concat([results_gdf, new_gdf], ignore_index=True)


                if enable_tiles:
                    save_prediction_tile(prediction, src.meta, prediction_tiles_dir, f'pred_{filename}')

    output_dict = {"data": results_gdf}
    if enable_tiles:
        os.makedirs(prediction_tiles_dir, exist_ok=True)
        create_file_list(prediction_tiles_dir, file_list_path)
        vrt_path = os.path.join(prediction_tiles_dir, 'predictions.vrt')
        png_path = os.path.join(prediction_tiles_dir, 'output.png')
        gdalbuildvrt(vrt_path, file_list_path)
        convert_vrt_to_png(vrt_path, png_path, width, height)
        output_dict.update({"vrt": vrt_path, "png": png_path})

    return output_dict








if __name__ == "__main__":


    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Generate predictions and visualizations for TIFF data.")
    parser.add_argument("data_dir", type=str, help="Directory containing TIFF files.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    args = parser.parse_args()


    map_directory = f'html_maps/{os.path.basename(args.data_dir)}'
    map_path = f'{map_directory}/vegetation_map.html'

    # Check if the map already exists
    if os.path.exists(map_path):
        print(f"Map already exists at {map_path}. Skipping generation.")
    else:
        # If the map does not exist, proceed with generating predictions and the map
        result = generate_predictions_and_visualizations(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir='output/' + os.path.basename(args.data_dir),
            enable_tiles=False
        )

        geodf_with_preds = result['data']

        # Visualization and saving the map as HTML
        cmap = plt.cm.Greens
        norm = Normalize(vmin=geodf_with_preds['prediction'].min(), vmax=geodf_with_preds['prediction'].max())
        greens_map = geodf_with_preds.explore(
            column='prediction',
            cmap='Greens',
            legend=True,
            legend_kwds={
                'label': "Percentage of Vegetation",
                'orientation': "horizontal"
            },
            style_kwds={'color': 'black', 'fillOpacity': 0.7, 'weight': 0.0}  # Set weight to 0 to hide borders

        )

        if not os.path.exists(map_directory):
            os.makedirs(map_directory)

        greens_map.save(map_path)
        print("Map saved successfully at " + map_path)