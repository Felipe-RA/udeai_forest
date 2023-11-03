import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpectralAttention import SpectralAttention

class VGGUdeaSpectral(nn.Module):
    def __init__(self, num_bands, 
                 num_filters1=64, 
                 num_filters2=128, 
                 num_filters3=256, 
                 activation_type='relu', 
                 dropout_rate=0.5,
                 fc1_out_features=1024,
                 fc2_out_features=512,
                 number_out_features=1):
        
        super(VGGUdeaSpectral, self).__init__()
        
        # Activation function selection
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation_type")
        
        self.conv_stage1 = nn.Sequential(
            nn.Conv2d(num_bands, num_filters1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters1),
            SpectralAttention(num_filters1),
            self.activation,
            nn.Conv2d(num_filters1, num_filters1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters1),
            SpectralAttention(num_filters1),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters2),
            SpectralAttention(num_filters2),
            self.activation,
            nn.Conv2d(num_filters2, num_filters2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters2),
            SpectralAttention(num_filters2),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_stage3 = nn.Sequential(
            nn.Conv2d(num_filters2, num_filters3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters3),
            SpectralAttention(num_filters3),
            self.activation,
            nn.Conv2d(num_filters3, num_filters3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters3),
            SpectralAttention(num_filters3),
            self.activation,
            nn.Conv2d(num_filters3, num_filters3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters3),
            SpectralAttention(num_filters3),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((12, 12))
        
        self.fc1 = nn.Sequential(
            nn.Linear(num_filters3 * 12 * 12, fc1_out_features),
            nn.BatchNorm1d(fc1_out_features),
            self.activation,
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_out_features, fc2_out_features),
            nn.BatchNorm1d(fc2_out_features),
            self.activation,
            nn.Dropout(dropout_rate)
        )
        
        self.output = nn.Linear(fc2_out_features, number_out_features)
        
    def forward(self, x):
        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x
