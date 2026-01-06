"""
Utility functions for the FER project.
"""
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Model Type: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("=" * 60 + "\n")


def get_device(preferred_device="cuda"):
    """
    Get the best available device.
    
    Args:
        preferred_device: Preferred device ('cuda' or 'cpu')
    
    Returns:
        device: torch device
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    return device


def save_experiment_config(config, save_path):
    """
    Save experiment configuration to a file.
    
    Args:
        config: ExperimentConfig object
        save_path: Path to save the configuration
    """
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Configuration saved to {save_path}")
