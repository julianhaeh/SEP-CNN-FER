"""
Configuration management for FER CNN experiments.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_dir: str = "data"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    image_size: int = 48
    num_classes: int = 7
    batch_size: int = 64
    num_workers: int = 4
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: int = 10
    horizontal_flip: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "BasicCNN"  # Options: BasicCNN, DeepCNN, VGG-like
    input_channels: int = 1  # Grayscale images
    num_classes: int = 7
    dropout_rate: float = 0.5
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # Options: adam, sgd, rmsprop
    scheduler: str = "step"  # Options: step, cosine, plateau
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    experiment_name: str = "fer_cnn_baseline"
    seed: int = 42
    device: str = "cuda"  # cuda or cpu
    log_dir: str = "logs"
    save_predictions: bool = True
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
