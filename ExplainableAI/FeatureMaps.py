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
from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced
from ModelArchitectures.clsMobileFaceNet import MobileFacenet
from ExplainableAI.GradCAM import OurGradCAM

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

class OurFeatureMaps:
    def __init__(self, model):
        self.model = model
    
    def FeatureMaps(self, img, target_layer = None):
        """
        shows first 16 feature maps of a chosen image and a chosen layer
        """
        layers = find_all_conv_layers(self.model)
        if target_layer is None:
            layer = layers[-1][1]
        else:
            layer = layers[target_layer][1]
        activations = {}
        def save_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        layer.register_forward_hook(save_activation("last_conv"))

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

    def get_top_feature_maps(self, image, target_layer = None, metric="mean"):
        """
        image: one tensor image
        layer: xth convolutional layer, if None then last layer
        metric: "mean" oder "max"
        shows top 16 feature maps of chosen metric
        """
        layers = find_all_conv_layers(self.model)
        if target_layer is None:
            layer = layers[-1][1]
        else:
            layer = layers[target_layer][1]
        features = {}

        def hook_fn(module, input, output):
            features['maps'] = output.detach().cpu()
        handle = layer.register_forward_hook(hook_fn)
        _ = self.model(image)
        handle.remove()
        maps = features['maps'][0]
        C = maps.shape[0]

        if metric == "mean":
            scores = maps.view(C, -1).mean(dim=1)
        elif metric == "max":
            scores = maps.view(C, -1).max(dim=1).values
        else:
            raise ValueError("Unknown metric")

        top_idx = torch.topk(scores, 16).indices
        top_maps = maps[top_idx]

        fig, ax = plt.subplots(4, 4, figsize=(8, 8))

        for i in range(16):
            ax[i // 4, i % 4].imshow(top_maps[i].cpu(), cmap="viridis")
            ax[i // 4, i % 4].axis("off")

        plt.tight_layout()
        plt.show()

        return top_maps, top_idx