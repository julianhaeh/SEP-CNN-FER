""""
This file defines the Self Curing Network (SCN) wrapper around our VGG13 architecture. 
The SCN comes from the paper "Suppressing Uncertainties for Large-Scale Facial Expression Recognition" by Wang et al. 2020.
In their paper they provided an python implementation, which file is heavily depended on. 
The SCN loss work by adding a self-attention mechanism to to the architecture, which can weight the models importance, thus weightening samples whose correct
label is unclear, either because of wrong labeling or ambiguous facial expression. It adds a rank regularization loss to the standard cross-entropy loss, 
which forces the model to give higher attention weights to "easy" samples, thus only eliminating the uncertain samples during training. In addition to that, 
the SCN relabels samples when it is really certain about its prediction, which can lead to the model cleaning up the wrong labeling.  
The SCN wrapper class is used in the experiment script pltSCNLoss.py.
"""

import torch.nn as nn
import torch
from torch.nn import init

from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced

NUM_CLASSES = 6  # Number of emotion classes


def weights_init(m):
    """Weight init for SGD, this stops gradient explosion or vanishing gradient"""
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)


class SCN_VGG_Wrapper(nn.Module):
    def __init__(self, base_model):
        
        super(SCN_VGG_Wrapper, self).__init__()

        # Reuse pretrained backbone 
        self.backbone = base_model.features 
        
        # New classifier
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

        # SCN Module
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        alpha_in_dim = 192
        self.alpha = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(alpha_in_dim, 1),
            nn.Sigmoid()
        )
        
        self.classifier.apply(weights_init)
        self.alpha.apply(weights_init)

    def forward(self, x):
        
        # Pass through pretrained backbone
        x = self.backbone(x)


        # Pass through attention module
        input_attention = self.avgpool(x) 
        input_attention = input_attention.view(input_attention.size(0), -1)
        attention_weights = self.alpha(input_attention)
        
        # Flatten x for classifier
        x = x.view(x.size(0), -1)
        raw_logits = self.classifier(x)
        
        # Multiply attention weights with classifier output
        out = attention_weights * raw_logits

        return attention_weights, raw_logits, out
    
if __name__ == "__main__":
    base_model = DownsizedCustomVGG13Reduced()
    model = SCN_VGG_Wrapper(base_model)
    print("Total parameters", sum(p.numel() for p in model.parameters()))

    testSamples = torch.randn(4, 1, 64, 64)  # Batch of 4 grayscale 64x64 images
    att_weights, raw_logits, out = model(testSamples)
    print("Attention weights shape:", att_weights.shape)
    print("Raw logits shape:", raw_logits.shape)
    print("Output shape:", out.shape)