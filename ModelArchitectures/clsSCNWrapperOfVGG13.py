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

NUM_CLASSES = 6 # Number of emotion classes

class SCN_VGG_Wrapper(nn.Module):
    def __init__(self, base_model):
        
        super(SCN_VGG_Wrapper, self).__init__()

        # Base model layers
        self.features = base_model.features 
        self.classifier = base_model.classifier

        # SCN Module
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Avg pool to reduce dimensionality
        alpha_in_dim = 256 # Input dimension after avg pool
        self.alpha = nn.Sequential(nn.Linear(alpha_in_dim, 1),nn.Sigmoid()) # Attention weight module

    def forward(self, x):

        # Pass through base model feature layer
        x = self.features(x)

        # Pass through attention module
        input_attention = self.avgpool(x) 
        input_attention = input_attention.view(input_attention.size(0), -1)
        attention_weights = self.alpha(input_attention)
        
        # Flatten x for classifier
        x = x.view(x.size(0), -1)
        
        # Multiply attention weights with classifier output
        out = attention_weights * self.classifier(x)

        # Return attention weights for rank regularization and final output for prediciton
        return attention_weights, out