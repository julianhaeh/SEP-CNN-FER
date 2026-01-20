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
    def __init__(self, base_model, drop_rate = 0):
        
        super(SCN_VGG_Wrapper, self).__init__()
        self.drop_rate = drop_rate
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = base_model.features 

        fc_in_dim = fc_in_dim = base_model.classifier[0].in_features# original fc layer's in dimention 512
   
        self.fc = nn.Linear(fc_in_dim, NUM_CLASSES) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        
        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out