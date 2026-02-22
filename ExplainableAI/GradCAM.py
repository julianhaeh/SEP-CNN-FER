import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from Data.clsOurDatasetSCN import OurDatasetSCN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDataset import OurDataset
from sklearn.decomposition import PCA

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def find_last_conv_index(model):
    last_idx = None
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d):
            last_idx = idx
    return last_idx

def find_all_conv_layers(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers

class OurGradCAM:
    def __init__(self, model):
        self.model = model
    
    def GradCAM(self, image, target_layer = None):
        """
        shows Grad-CAM of one input image, with a chosen layer (if None: last layer)
        """
        layers = find_all_conv_layers(self.model)
        if target_layer is None:
            layer = layers[-1][1]
        else:
            layer = layers[target_layer][1]
        cam = GradCAM(self.model, target_layers = [layer])
        grayscale_cam = cam(input_tensor=image)[0]
        base_img = image[0].cpu().numpy()
        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
        rgb_img = np.stack([base_img]*3, axis=-1)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.show()
        return visualization, target_layer

    def kGradCAMs(self, images, target_layer = None):
        """
        shows Grad-CAMs for list of k images with a chosen layer (if None: last layer)
        """
        k = len(images)
        k = k - (k % 4)
        layers = find_all_conv_layers(self.model)
        if target_layer is None:
            layer = layers[-1][1]
        else:
            layer = layers[target_layer][1]
        cam = GradCAM(self.model, target_layers = [layer])
        GradCAMs = []
        for i in range (k):
            grayscale_cam = cam(input_tensor=images[i])[0]
            base_img = images[i][0].cpu().numpy()
            base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
            rgb_img = np.stack([base_img]*3, axis=-1)[0]
            GradCAMs.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))
        fig, axes = plt.subplots(4, int(k/4))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(GradCAMs[idx])
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()
