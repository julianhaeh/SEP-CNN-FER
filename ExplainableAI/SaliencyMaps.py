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

class OurSaliencyMaps:
    def __init__(self, model):
        self.model = model
    
    def SaliencyMap(self, img):
        imgshow = img.squeeze(0).permute(1, 2, 0).numpy()
        img.requires_grad = True
        output = self.model(img)
        score = output[0, output.argmax()]
        score.backward()
        saliency = img.grad.data.abs().squeeze()
        plt.imshow(imgshow, cmap="gray")
        plt.imshow(saliency, cmap="hot", alpha=0.55)
        plt.axis("off")
        plt.show()
        img.requires_grad = False
        return saliency