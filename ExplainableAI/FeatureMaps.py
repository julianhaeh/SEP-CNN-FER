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
from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced
from ModelArchitectures.clsMobileFaceNet import MobileFacenet
from ExplainableAI.GradCAM import OurGradCAM

def find_last_conv_index(model):
    last_idx = None
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d):
            last_idx = idx
    return last_idx

class OurFeatureMaps:
    def __init__(self, model):
        self.model = model
    
    def FeatureMaps(self, img, target_layer = None):
        if target_layer is None:
            target_layer = find_last_conv_index(self.model)
        activations = {}
        def save_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        self.model.features[target_layer].register_forward_hook(save_activation("last_conv"))

        _ = self.model(img)
        feat = activations["last_conv"][0]

        num_maps = 16
        cols = 4
        rows = num_maps // cols

        fig, ax = plt.subplots(rows, cols, figsize=(8, 8))

        for i in range(num_maps):
            ax[i // cols, i % cols].imshow(feat[i].cpu(), cmap="viridis")
            ax[i // cols, i % cols].axis("off")

        plt.tight_layout()
        plt.show()