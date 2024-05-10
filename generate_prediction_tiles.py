import argparse
import torch
import joblib
import numpy as np
import rasterio
import os
import json
import logging
from tqdm import tqdm
from osgeo import gdal
from sklearn.metrics import mean_squared_error



# Setup a specific logger for this script
logger = logging.getLogger('PredictionTileLogger')
logger.setLevel(logging.INFO)  # Default level

# Create handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(handler)

# Custom model import
from src.classes.VGGUdeaSpectral import VGGUdeaSpectral

def rmse_score(net, X, y):
    y_pred = net.predict(X)
    rmse = (mean_squared_error(y_true=y, y_pred=y_pred)) ** 0.5
    return -rmse  # Skorch tries to maximize the score, so negate the RMSE

def parse_args():
    parser = argparse.ArgumentParser(description="Process satellite images to generate prediction tiles.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing TIFF files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions and metadata.")
    parser.add_argument("--output_width", type=int, default=2560, help="Width of the output PNG image.")
    parser.add_argument("--output_height", type=int, default=1440, help="Height of the output PNG image.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output.")
    return parser.parse_args()

def read_tif_file(filepath):
    logger.debug(f"Reading TIFF file: {filepath}")
    with rasterio.open(filepath) as src:
        return np.array(src.read()), src.meta

def save_prediction_tile(prediction, metadata, output_dir, filename):
    logger.debug(f"Saving prediction tile: {os.path.join(output_dir, filename)}")
    prediction_tile = np.full((100, 100), prediction, dtype=np.float32)
    new_meta = metadata.copy()
    new_meta.update({
        'dtype': 'float32',
        'count': 1,
        'driver': 'GTiff'
    })
    with rasterio.open(os.path.join(output_dir, filename), 'w', **new_meta) as dest:
        dest.write(prediction_tile, 1)

def save_metadata(metadata_list, save_path):
    logger.debug(f"Saving metadata to: {save_path}")
    for meta in metadata_list:
        if 'sentinel_meta' in meta and 'crs' in meta['sentinel_meta']:
            meta['sentinel_meta']['crs'] = str(meta['sentinel_meta']['crs'])
    with open(save_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)

def create_file_list(directory, output_file):
    logger.debug(f"Creating file list at: {output_file}")
    with open(output_file, 'w') as file_list:
        for filename in os.listdir(directory):
            if filename.endswith('.tif'):
                file_list.write(os.path.join(directory, filename) + '\n')

def convert_vrt_to_png(vrt_path, png_path, output_width, output_height):
    logger.debug(f"Converting VRT to PNG: {png_path} with resolution {output_width}x{output_height}")
    dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    min, max = band.ComputeRasterMinMax()
    options = gdal.TranslateOptions(format='PNG', outputType=gdal.GDT_Byte,
                                    width=output_width, height=output_height,
                                    scaleParams=[[min, max, 0, 255]])
    gdal.Translate(png_path, dataset, options=options)
    dataset = None

def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)  # Set logging to debug level if --debug is set

    os.makedirs(args.output_dir, exist_ok=True)

    vgg_model = joblib.load(args.model_path)
    metadata_list = []
    file_list_path = os.path.join(args.output_dir, 'file_list.txt')
    
    for filename in tqdm(os.listdir(args.data_dir)):
        if filename.endswith('.tif'):
            sentinel_path = os.path.join(args.data_dir, filename)
            sentinel_img, sentinel_meta = read_tif_file(sentinel_path)
            image_tensor = torch.tensor(sentinel_img, dtype=torch.float32).unsqueeze(0)
            prediction = vgg_model.predict(image_tensor)
            save_prediction_tile(prediction, sentinel_meta, args.output_dir, f'pred_{filename}')
            metadata_list.append({'filename': f'pred_{filename}', 'sentinel_meta': sentinel_meta})
    
    save_metadata(metadata_list, os.path.join(args.output_dir, 'metadata.json'))
    create_file_list(args.output_dir, file_list_path)

    vrt_command = f"gdalbuildvrt -input_file_list {file_list_path} {os.path.join(args.output_dir, 'predictions.vrt')}"
    os.system(vrt_command)
    convert_vrt_to_png(os.path.join(args.output_dir, 'predictions.vrt'), 
                       os.path.join(args.output_dir, 'output.png'), 
                       args.output_width, args.output_height)

if __name__ == "__main__":
    main()
