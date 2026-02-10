import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

class OurGradCAM:
    def __init__(self, model):
        self.model = model
    
    def GradCAM(self, image, target_layer = None):
        if target_layer is None:
            target_layer = find_last_conv_index(self.model)
        layer = self.model.features[target_layer]
        cam = GradCAM(self.model, target_layers = [layer])
        grayscale_cam = cam(input_tensor=image)[0]
        base_img = image[0].cpu().numpy()
        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
        rgb_img = np.stack([base_img]*3, axis=-1)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.show()
        return visualization, target_layer