"""
This file defines a CustomCNN class, which allows for flexible definition of CNN architectures. For details on how to initialize the architectures, 
see the documentation of the __init__ method.
"""



import torch
import torch.nn as nn

INPUT_SHAPE = (1, 64, 64)  # Hardcoded input shape for our data

class CustomCNN(nn.Module):
    
    def __init__(self, feature_config, classifier_config):
        """
        The custom CNN architecture is defined by the lists of dictionaries `feature_config` and `classifier_config`.
        Each dictionary in these lists specifies a layer type and its parameters. The architecture is fixed to first 
        build the feature extraction part (convolutional layers) followed by the classification part (fully connected layers).

        Possible inputs for `feature_config`:
            - Convolutional Layer: {'type': 'conv', 'out': int, 'k': int, 's': int, 'p': int}
                - 'out': number of output channels
                - 'k': kernel size (default 3)
                - 's': stride (default 1)
                - 'p': padding (default 1)
            - Activation Layer: {'type': 'act'}
            - Pooling Layer: {'type': 'pool'}
                - 'k': kernel size (default 2)
                - 's': stride (default 2)
            - Dropout Layer: {'type': 'dropout', 'p': float}
                - 'p': dropout probability
            - Batch Normalization Layer: {'type': 'norm'}
            - Global Average Pooling Layer: {'type': 'gap'}

        Possible inputs for `classifier_config`:
            - Fully Connected Layer: {'type': 'full', 'out': int}
                - 'out': number of output features
            - Activation Layer: {'type': 'act'}
            - Dropout Layer: {'type': 'dropout', 'p': float}
                - 'p': dropout probability
            - Batch Normalization Layer: {'type': 'norm'}

        There is no need to define the input dimension of the layers, as it is inferred from the previous layers and the hardcoded input shape.
        Note that the last layer of the classifier should have an output dimension of 6 for our 6 emotion classes.
        
        Example usage: 

            from ModelArchitectures.clsCustomCNN import CustomCNN

            feature_config = [ 
            {'type': 'conv', 'out': 16, 'k': 3, 's': 1, 'p': 1},
            {'type': 'act'},
            {'type': 'pool'},
            {'type': 'conv', 'out': 32, 'k': 3, 's': 1, 'p': 1},
            {'type': 'pool'}
            ]

            classifier_config = [
                {'type' : 'full', 'out': 128},
                {'type': 'act'},
                {'type': 'full', 'out': 6}
            ]

            CustomCNNModel = CustomCNN(feature_config, classifier_config)
        
        """
        super().__init__()
        
        self.features = nn.Sequential()
        current_channels = INPUT_SHAPE[0] # Since INPUT_SHAPE is hardcoded, this will be 1. The current_channels variable tracks the input channel dimension for the next layer.
        
        # Loop for the convolutional, pooling layers
        for i, config in enumerate(feature_config):
            
            if config['type'] == 'conv':
                self.features.add_module(f"conv_{i}", nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=config['out'],
                    kernel_size=config.get('k', 3),
                    stride=config.get('s', 1),
                    padding=config.get('p', 1),
                ))
                current_channels = config['out'] # Update for next layer
                
            elif config['type'] == 'act':
                self.features.add_module(f"act_{i}", nn.ReLU(inplace=True))
                
            elif config['type'] == 'pool':
                    size = config.get('k', 2)
                    stride = config.get('s', 2)
                    self.features.add_module(f"pool_{i}", nn.MaxPool2d(size, stride))

            elif config['type'] == 'norm': 
                self.features.add_module(
                    f"norm_{i}",
                    nn.BatchNorm2d(current_channels)
                )

            elif config['type'] == 'gap':
                self.features.add_module(
                    f"gap_{i}",
                    nn.AdaptiveAvgPool2d((1, 1))
                )
            
            elif config['type'] == 'dropout':
                self.features.add_module(f"drop_{i}", nn.Dropout2d(config['p']))

        # Perform one dummy pass through the layers from before to determine the input channel dimension for the first fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *INPUT_SHAPE) 
            feature_out = self.features(dummy_input)    
            flat_dim = feature_out.view(1, -1).size(1) 
            
        self.classifier = nn.Sequential()
        
        # Add a flattening layer before the fully connected layers
        self.classifier.add_module("flatten", nn.Flatten())
        
        current_in_features = flat_dim 
        
        for i, config in enumerate(classifier_config):
            if config['type'] == 'full':
                self.classifier.add_module(f"fc_{i}", nn.Linear(
                    in_features=current_in_features, 
                    out_features=config['out']
                ))
                current_in_features = config['out'] # Update for next layer
                
            elif config['type'] == 'act':
                self.classifier.add_module(f"act_fc_{i}", nn.ReLU(inplace=True))
                
            elif config['type'] == 'dropout':
                 self.classifier.add_module(f"drop_{i}", nn.Dropout(config['p']))

            elif config['type'] == 'norm':
                self.classifier.add_module(
                    f"norm_fc_{i}",
                    nn.BatchNorm1d(current_in_features)
                )

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out