import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import init
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

from ModelArchitectures.clsCustomCNN import CustomCNN 
from Data.clsOurDataset import OurDataset

# --- EXPERIMENT CONSTANTS ---
EPOCHS = 55
BATCH_SIZE = 32
USE_SCHEDULER = True

# --- DEBUG CONSTANTS ---
SAVE_MODELS = False
LOG_FILE = "Experiments/Plots/custom_cnn_training_history.txt"



# Class mappings
EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

# Global Class Weights (for Weighted CE)
CLASS_WEIGHTS_TENSOR = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

# Basline configs of original CustomVGG13Reduced

original_feature_config = [
                                # --- Block 1 ---
                                {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},  
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 2 ---
                                {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 3 ---
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 4 ---
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},

                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},
                            ]

original_classifier_config = [
                                
                                {'type': 'full', 'out': 1024},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'dropout', 'p': 0.5},
                                
                                {'type': 'full', 'out': 1024},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'dropout', 'p': 0.5},
                                
                                {'type': 'full', 'out': 6} 
                            ]

MODEL_CONFIGS = {

    "VGG13_Original" : {
            "feature_config": original_feature_config,
            "classifier_config": original_classifier_config
    },

    "VGG13_ExtraBlockInFeatureLayer_A" : {
            "feature_config": [
                                # --- Block 1 ---
                                {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},  
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 2 ---
                                {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 3 ---
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 4 ---
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},

                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 5 --- (Newly Added)

                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},

                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25}

                            ],
            "classifier_config": original_classifier_config
                        
    },

    "VGG_Removed1BlockInFeatureLayer_B" : {
            "feature_config": [
                                # --- Block 1 ---
                                {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},  
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 2 ---
                                {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25},

                                # --- Block 3 ---
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                
                                {'type': 'pool'},
                                {'type': 'dropout', 'p': 0.25}

                                # Blocks 4 removed

                            ],
            "classifier_config": original_classifier_config

    },

    "VGG13_ExtraLayer_Classifier_F" : {
            "feature_config": original_feature_config,
            "classifier_config": [
                                
                                {'type': 'full', 'out': 1024},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'dropout', 'p': 0.5},
                                
                                {'type': 'full', 'out': 1024},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'dropout', 'p': 0.5},

                                {'type': 'full', 'out': 256},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'dropout', 'p': 0.5},

                                {'type': 'full', 'out': 6}
                            ]
    },

    "VGG13_Removed1Layer_Classifier_E" : {

            "feature_config": original_feature_config,
            "classifier_config": [
                                {'type': 'full', 'out': 1024},
                                {'type': 'norm'}, # New Layer
                                {'type': 'act'},
                                {'type': 'dropout', 'p': 0.5},
                                {'type': 'full', 'out': 6}
                            ]
    },

    "VGG13_AddedOneLayerPerBlock_Feature_C" : {

                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},  # New Layer
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},  # New Layer
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},

                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},  # New Layer
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 4 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},  # New Layer
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},

                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25}
                                ],
                "classifier_config": original_classifier_config
    },

    "VGG13_RemovedOneLayerPerBlock_Feature_D" : {
                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 4 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},

                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25}
                                ],
                "classifier_config": original_classifier_config
    },

    "VGG13_B_And_HalfClassifier" : {

                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25}

                                    # Blocks 4 removed"
                                ],

                "classifier_config": [
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 6}
                                ]
    },

    "VGG13_B_And_HalfClassifier_And_Gap" : {

                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},
                                    {'type': 'gap'}

                                    # Blocks 4 removed"
                                ],

                "classifier_config": [
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 6}
                                ]
    }, 

    "VGG13_B_And_HalfClassifier_StrongPooling" : {

                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25}

                                    # Blocks 4 removed"
                                ],

                "classifier_config": [
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 6}
                                ]
    },

    "VGG13_B_And_FourthClassifier" : {

                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25}

                                    # Blocks 4 removed"
                                ],

                "classifier_config": [
                                    
                                    {'type': 'full', 'out': 256},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 256},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 6}
                                ]
    },

    "VGG13_B_And_HalfClassifier_KernelCap192" : {

                "feature_config": [# --- Block 1 ---
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},  
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 2 ---
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25},

                                    # --- Block 3 ---
                                    {'type': 'conv', 'out': 192, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 192, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'conv', 'out': 192, 'k': 3, 's': 1, 'p': 1},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    
                                    {'type': 'pool'},
                                    {'type': 'dropout', 'p': 0.25}

                                    # Blocks 4 removed"
                                ],

                "classifier_config": [
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 512},
                                    {'type': 'norm'}, # New Layer
                                    {'type': 'act'},
                                    {'type': 'dropout', 'p': 0.5},
                                    
                                    {'type': 'full', 'out': 6}
                                ]
    }




}

MODELS_TO_TRAIN = [
    #( "VGG13_Original", MODEL_CONFIGS["VGG13_Original"]),
    #( "VGG13_ExtraBlockInFeatureLayer_A", MODEL_CONFIGS["VGG13_ExtraBlockInFeatureLayer_A"]),
    #( "VGG_Removed2BlocksInFeatureLayer_B", MODEL_CONFIGS["VGG_Removed2BlocksInFeatureLayer_B"]),
    #( "VGG13_AddedOneLayerPerBlock_Feature_C", MODEL_CONFIGS["VGG13_AddedOneLayerPerBlock_Feature_C"]),
    #( "VGG13_RemovedOneLayerPerBlock_Feature_D", MODEL_CONFIGS["VGG13_RemovedOneLayerPerBlock_Feature_D"]),
    #( "VGG13_Removed1Layer_Classifier_E", MODEL_CONFIGS["VGG13_Removed1Layer_Classifier_E"]),
    #( "VGG13_ExtraLayer_Classifier_F", MODEL_CONFIGS["VGG13_ExtraLayer_Classifier_F"]),
    #( "VGG13_B_And_SmallerClassifier", MODEL_CONFIGS["VGG13_B_And_SmallerClassifier"])
    #( "VGG13_B_And_HalfClassifier_And_Gap", MODEL_CONFIGS["VGG13_B_And_HalfClassifier_And_Gap"]),
    #( "VGG13_B_And_HalfClassifier_StrongPooling", MODEL_CONFIGS["VGG13_B_And_HalfClassifier_StrongPooling"])
    #( "VGG13_B_And_FourthClassifier", MODEL_CONFIGS["VGG13_B_And_FourthClassifier"])
    ( "VGG13_B_And_HalfClassifier_KernelCap192", MODEL_CONFIGS["VGG13_B_And_HalfClassifier_KernelCap192"])
]
# --- HELPER FUNCTIONS ---

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = torch.tensor([], dtype=torch.long)
    all_labels = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            out = model(inputs)
            _, preds = torch.max(out, 1)
            all_preds = torch.cat((all_preds, preds.cpu()))
            all_labels = torch.cat((all_labels, labels.cpu()))
    return all_labels, all_preds

def compute_confusion_matrix_torch(true_labels, pred_labels, num_classes=6):
    indices = true_labels * num_classes + pred_labels
    cm = torch.bincount(indices, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes).float()

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_history(history, title, filename, ylabel="Value"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

# --- TRAINING PIPELINE ---

def train_evaluate_pipeline(model_config, config_name=""):
    """
    Runs a full training session for one specific configuration.
    Returns loss_history, accuracy_history, and final predictions.
    """
    # 1. Unpack Architecture for this run
    feature_config = model_config["feature_config"]
    classifier_config = model_config["classifier_config"]

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainDataLoader = DataLoader(OurDataset(split='train'), batch_size=BATCH_SIZE, shuffle=True)
    valDataLoader = DataLoader(OurDataset(split='test'), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model & Weights
    model = CustomCNN(feature_config, classifier_config)
    model.apply(weights_init)
    model.to(device)

    print(f"Model {config_name} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    # 4. Initialize Optimizer (Fixed SGD)
    optimizer = optim.SGD(model.parameters(), lr=0.014, momentum=0.9, weight_decay=2.2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
        
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS_TENSOR.to(device))

    train_loss_history = []
    acc_history = []
    test_loss_history = []
    
    # --- EPOCH LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(trainDataLoader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
    
        for batch in train_loop:
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            out = model(imgs)
            
            loss = criterion(out, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
            
        # Record training metrics
        train_loss_history.append(running_loss / len(trainDataLoader))
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in valDataLoader:
                imgs = batch['image'].to(device)
                targets = batch['label'].to(device)
                
                out = model(imgs)
                loss = criterion(out, targets) # Calculate Test Loss
                
                val_running_loss += loss.item()
                
                # Calculate Accuracy
                _, preds = torch.max(out, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        val_epoch_loss = val_running_loss / len(valDataLoader)
        val_epoch_acc = val_correct / val_total * 100
        
        test_loss_history.append(val_epoch_loss)
        acc_history.append(val_epoch_acc)
        
        print(f"   [Epoch {epoch+1}] Train Loss: {train_loss_history[-1]:.4f} | Test Loss: {val_epoch_loss:.4f} | Test Acc: {val_epoch_acc:.2f}%")

    # Get final predictions for Confusion Matrix
    y_true_final, y_pred_final = get_all_predictions_torch(model, valDataLoader, device)
    
    return model, train_loss_history, test_loss_history, acc_history, y_true_final, y_pred_final


# --- MAIN EXPERIMENT LOOP ---

def run_experiments():
    print("Starting Custom CNN Experiments...")
    
    for config_name, config_model in MODELS_TO_TRAIN:

        print(f"\n==============================================")
        print(f"Running Experiment: {config_name}")
        print(f"==============================================")
            
        # 1. FIXED UNPACKING: Now accepts 6 variables
        model, train_loss, test_loss, acc_hist, y_true, y_pred = train_evaluate_pipeline(
            config_model, 
            config_name=config_name
        )
            
        # 2. Plot Training Loss
        plot_history(
            train_loss, 
            f"Training Loss {config_name}", 
            f"Experiments/Plots/Train_Loss_{config_name}.png", 
            ylabel="Loss"
        )

        # 3. Plot Test Loss 
        plot_history(
            test_loss, 
            f"Test Loss {config_name}", 
            f"Experiments/Plots/Test_Loss_{config_name}.png", 
            ylabel="Loss"
        )
            
        # 4. Plot Test Accuracy History
        plot_title_acc = f"Test Accuracy {config_name}"
        filename_acc = f"Experiments/Plots/Accuracy_{config_name}.png"
        plot_history(acc_hist, plot_title_acc, filename_acc, ylabel="Accuracy (%)")
            
        # 5. Plot Confusion Matrix
        cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)
        plot_title_cm = f"Confusion Matrix {config_name}"
        filename_cm = f"Experiments/Plots/CM_{config_name}.png"

        plot_confusion_matrix(
            cm_tensor.numpy(), 
            CLASS_NAMES, 
            plot_title_cm, 
            filename_cm
        )

        if SAVE_MODELS:
            torch.save(model.state_dict(), f"Experiments/Models/{config_name}_Acc{int(acc_hist[-1])}.pth")

        with open(LOG_FILE, "a") as f:
            f.write(f"Experiment: {config_name}, Accuracy: {acc_hist[-1]:.2f}%, Test-Loss: {test_loss[-1]:.4f}, Train-Loss: {train_loss[-1]:.4f}\n")
            
        print(f"Completed {config_name}. Plots saved.")


if __name__ == "__main__":
    run_experiments()