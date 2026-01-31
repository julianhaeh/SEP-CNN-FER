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
from torch.nn import init

from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced

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
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, NUM_CLASSES)
        )

        # SCN Module
        alpha_in_dim = 256
        self.alpha = nn.Sequential(
            nn.BatchNorm2d(alpha_in_dim),  
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(alpha_in_dim, 1),
            nn.Sigmoid()
        )

        
        
        # Initialize new layers
        self.classifier.apply(weights_init)
        self.alpha.apply(weights_init)

        # Freeze backbone parameters
        # for param in self.backbone.parameters():
        #    param.requires_grad = False

    def forward(self, x):
        
        # Pass through pretrained backbone
        x = self.backbone(x)


        # Pass through attention module
        attention_weights = self.alpha(x)
        
        # Flatten x for classifier
        x = x.view(x.size(0), -1)
        raw_logits = self.classifier(x)
        
        # Multiply attention weights with classifier output
        out = attention_weights * raw_logits

        return attention_weights, raw_logits, out
    
if __name__ == "__main__":
    base_model = CustomVGG13Reduced()
    model = SCN_VGG_Wrapper(base_model)
    print("Total parameters", sum(p.numel() for p in model.parameters()))