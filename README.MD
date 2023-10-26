# Análisis y predicción de cobertura vegetal en Antioquia con Imágenes satelitales de Sentinel-2

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Dependencies](#dependencies)
- [Data Collection](#data-collection)
  - [Security Considerations](#security-considerations)
  - [Generate Grid](#generate-grid)
  - [Download Data](#download-data)

## Introduction

This project aims to generate predictions of the percentage of vegetation cover in Antioquia based on the Sentinel-2 and MOD44B.006 Terra Vegetation datasets. The architecture is designed to accept any number of bands (`num_bands`), but defaults to three bands corresponding to the RGB channels (B4, B3, B2) of Sentinel-2. The Sentinel-2 bands serve as the input to our model, while the target variable (total vegetation cover) is obtained by summing the `Percent_Tree_Cover` and `Percent_NonTree_Vegetation` bands to get `Percent_VegetationCover`.

## Motivation

The motivation behind this project is to provide government and environmental organizations with tools to predict vegetation coverage based on satellite images. This allows for better environmental monitoring and decision-making.

## Installation

### System Requirements

- Officially tested on Ubuntu 22.04 LTS.
- Other operating systems and distributions may work but are not officially supported.
- Windows, as of October 2023, does not have the necessary dependencies to train the models on AMD GPUs.

### Dependencies

To run the project, you'll need:

- Python 3.x
- pip
- A modified version of the `geetiles` tool, which can be installed using the following command:

```bash
pip install git+https://github.com/Felipe-RA/geetiles
```

## Data Collection

### Security Considerations

**BEWARE! BEFORE RUNNING THE DOWNLOAD COMMAND:**

We use a Notebook Authenticator for the project. Google will likely issue a warning advising you not to run the code unless you are using a notebook and understand the code being executed. The code responsible for these operations is located in the `defs` directory. This code defines the instruments (ImageCollection) and bands to be used, among other post-processing steps. While our `defs` files do not access your private data, malicious applications could misuse token access to compromise your information. **Make sure to read and understand the code before executing any commands.**

### Generate Grid

Use the `geet` tool to generate a grid over your Area of Interest (AOI) with the following command:

```bash
geet grid --aoi_wkt_file data/antioquia.wkt  --chip_size_meters 1000 --aoi_name antioquia --dest_dir .
```

- `--aoi_wkt_file`: Path to the `.wkt` file defining your AOI.
- `--chip_size_meters`: Size in meters of each chip.
- `--aoi_name`: Name for the `.geojson` file to be created.
- `--dest_dir`: Directory where the `.geojson` file will be saved.

### Download Data

**Important**: You need a Google Earth Engine account and a cloud project to use the Earth Engine data pipeline.

1. **Authorization**: We use a Notebook Authenticator. Be cautious when following Google's warnings. The code that performs these operations is in the `defs` directory.
  
2. **Download Duration**: Depending on various factors, the download can take from minutes to days. For the `antioquia.wkt` AOI with 1000-meter chips, expect around 3 to 5 hours.

Execute the following command to start the download:

```bash
geet download --tiles_file path/to/your_grid_partition_aschips.geojson --dataset_def defs/CHOOSE_DEF_FILE.py --pixels_lonlat [100,100] --skip_if_exists
```

Replace `CHOOSE_DEF_FILE.py` with your definition file like `treecover2020.py` or `sentinel2-rgb-median-2020`.

## Machine Learning Models

### Drivers and Toolkits

1. Install the drivers `CUDA Toolkit` for NVIDIA GPUs.
2. Install the `ROCm OpenSource Platform` for AMD GPUs with **SUPPORTED ARCHITECTURES**

### Installation of ML Dependencies

Be careful, **GPU accelerated tools** implementations are extremely dependant on the OS on which they are installed, all of the console commands listed below assume that you are working on **Linux**. If you are using other OS please refer to the [Official PyTorch Installation Matrix](https://pytorch.org/get-started/locally/)

To install the required dependencies for running the machine learning models, execute the following commands:

#### General requirements

```bash
pip install -r requirements.txt
```

#### If you will be using CUDA 11.8 (better with Turing series cards and earlier)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### If you will be using CUDA 12.1+ (better with Ampere series cards and later)

```bash
pip3 install torch torchvision torchaudio
```

#### **NEW!** AMD Support for PyTorch using ROCm [(Supported GPUs are changing on a regular basis, check here)](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html#linux-supported-gpus)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### Architecture

The models are compatible with PyTorch's `nn` module. They are inspired by the VGG architecture and are hence named VGGUdeA.

### Implementation

The specific implementation of the models can be found in the `src/classes` directory.

## Training Cycles and DataLoader Pipelines

### Available Tests

Various test configurations are available under `src/synthetic_test_training_cycles`. These tests use:

- GPU, if available and configured (more on GPU configurations later). CPU is used otherwise.
- Synthetic data generated specifically for testing the architecture without having to load and process the large `.tif` images as discussed in the [Data Download Section](#download-data).
- Multiple `test_*.py` files, each implementing different model optimization techniques.

### Performance Considerations

Be aware that running these tests on a CPU is considerably slower. What might take seconds on a GPU could take minutes on a CPU.

### DataLoader Pipeline for Satellital Imagery

**under construction...**

#### Raster processing using Rasterio

**UNDER CONSTRUCTION...**

## Dockerization

### Volumes and Data

**UNDER CONSTRUCTION...**

### Storing Models

**under construction...**

### Adding Stored Models

**under construction...**

### Running predictions from the Docker Container

**under construction...**
