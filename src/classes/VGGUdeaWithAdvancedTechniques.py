import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpectralAttention import SpectralAttention


class VGGUdeaWithAdvancedTechniques(nn.Module):
    def __init__(self, num_bands, dropout_rate=0.5, l1_factor=1e-5, l2_factor=1e-4):
        super(VGGUdeaWithAdvancedTechniques, self).__init__()
        
        self.spectral_attention = SpectralAttention(num_bands)
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 12 * 12, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc3 = nn.Linear(512, 1)
        
        # Regularization factors
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

    def forward(self, x):
        x = self.spectral_attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Adaptive pooling
        x = F.adaptive_avg_pool2d(x, (12, 12))
        
        # Flatten tensor
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

    def calculate_regularization_loss(self):
        l1_reg = sum(param.abs().sum() for param in self.parameters())
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
        
        return self.l1_factor * l1_reg + self.l2_factor * l2_reg