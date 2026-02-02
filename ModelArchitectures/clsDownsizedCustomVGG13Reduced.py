"""
This file defines a CustomVGG13 architecture, as it was introcuded in the paper "Training Deep Networks for Facial Expression Recognition
with Crowd-Sourced Label Distribution" by Barsoum et al. 2016. We chose this architecture since it has only around 7.7 million parameters, 
which is small enough to be train on our rather small dataset without overfitting too much, while still being a deep architecture. 
"""
import torch
import torch.nn as nn

INPUT_SHAPE = (1, 64, 64)  # Hardcoded input shape for our data

class DownsizedCustomVGG13Reduced(nn.Module):
    """
    A reduced version of the custom VGG13 architecture introduced in the paper "Training Deep Networks for Facial Expression Recognition
    with Crowd-Sourced Label Distribution", designed for smaller datasets and lower computational resources".
    The existing architecture is reduced to have less parameters, so it will not overfit on our rather small dataset and adjusted to our number of classes.
    """
    
    def __init__(self):

    
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Added BatchNorm
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Block 3
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),  # Added BatchNorm
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(192 * 8 * 8, 512),  # Assuming input images are 64x64
            nn.BatchNorm1d(512),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 6)
        )
        
    def forward(self, x):
        features = self.features(x)



        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits 

if __name__ == "__main__":
    model = DownsizedCustomVGG13Reduced()
    print("Total parameters", sum(p.numel() for p in model.parameters()))