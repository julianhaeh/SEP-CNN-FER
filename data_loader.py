"""
Data loading and preprocessing for FER datasets.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FERDataset(Dataset):
    """Custom dataset for Facial Expression Recognition."""
    
    def __init__(self, data_dir, csv_file, transform=None, image_size=48):
        """
        Args:
            data_dir: Directory containing the images
            csv_file: CSV file with annotations (columns: image_path, label)
            transform: Optional transform to be applied on images
            image_size: Size to resize images to
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load CSV file
        if os.path.exists(csv_file):
            self.data_frame = pd.read_csv(csv_file)
        else:
            # Create dummy data if CSV doesn't exist (for testing)
            print(f"Warning: {csv_file} not found. Creating dummy dataset.")
            self.data_frame = pd.DataFrame({
                'image_path': [],
                'label': []
            })
        
        # Emotion labels (standard FER2013 format)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if len(self.data_frame) == 0:
            # Return dummy data
            image = torch.zeros(1, self.image_size, self.image_size)
            label = 0
            return image, label
        
        img_name = os.path.join(self.data_dir, self.data_frame.iloc[idx]['image_path'])
        label = int(self.data_frame.iloc[idx]['label'])
        
        # Load image
        if os.path.exists(img_name):
            image = Image.open(img_name).convert('L')  # Convert to grayscale
        else:
            # Create dummy grayscale image if file doesn't exist
            image = Image.new('L', (self.image_size, self.image_size), color=128)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        return image, label


def get_transforms(image_size=48, augment=True):
    """
    Get data transforms for training and validation.
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        train_transform, val_transform
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform


def get_dataloaders(config):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: ExperimentConfig object
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_transform = get_transforms(
        image_size=config.data.image_size,
        augment=config.data.use_augmentation
    )
    
    # Create datasets
    train_dataset = FERDataset(
        data_dir=config.data.data_dir,
        csv_file=os.path.join(config.data.data_dir, config.data.train_csv),
        transform=train_transform,
        image_size=config.data.image_size
    )
    
    val_dataset = FERDataset(
        data_dir=config.data.data_dir,
        csv_file=os.path.join(config.data.data_dir, config.data.val_csv),
        transform=val_transform,
        image_size=config.data.image_size
    )
    
    test_dataset = FERDataset(
        data_dir=config.data.data_dir,
        csv_file=os.path.join(config.data.data_dir, config.data.test_csv),
        transform=val_transform,
        image_size=config.data.image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
