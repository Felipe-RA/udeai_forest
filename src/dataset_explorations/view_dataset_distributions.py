import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_data_distribution(X, y, num_bins=20):
    # Create subplots for Sentinel-2 data and target data
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))

    # Plot histograms for each band (R, G, B)
    colors = ['red', 'green', 'blue']
    band_labels = ['Band R', 'Band G', 'Band B']
    for i in range(3):
        axes[i].hist(X[:, i, :, :].ravel(), bins=100, color=colors[i], alpha=0.7)
        axes[i].set_title(r'$\bf{SENTINEL-2\ Satellital\ Imagery\ for\ Antioquia}$' + '\n' + 'Distribution of band color' + f'\n{band_labels[i]}')
        axes[i].set_xlabel('Pixel Values')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[3].hist(y, bins=100, color='purple', alpha=0.7)
    axes[3].set_title(r'$\bf{Distribution\ of\ Target Data\ (y)}$' + '\nPercentage of Vegetation Cover')
    axes[3].set_xlabel('Percentage Values')
    axes[3].set_ylabel('Frequency')
    axes[3].grid(True, linestyle='--', alpha=0.6)

    # Add a tight layout
    plt.tight_layout()

    # Show the plots
    plt.show()

# Assuming you have X_tensor and y_tensor as your data
X = torch.load('X_tensor.pth').numpy()
y = torch.load('y_tensor.pth').numpy()


# Print each unique value along with its count

visualize_data_distribution(X, y)
