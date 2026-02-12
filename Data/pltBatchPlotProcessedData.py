"""
This script generates and saves a batch plot of processed images from our dataset.
"""
import matplotlib.pyplot as plt
from datasets import load_from_disk
import random
from clsOurDataset import OurDataset
from torch.utils.data import DataLoader
from Data.clsOurDatasetTuning import OurDatasetTuning
from torchvision.transforms import v2
import numpy as np

label_map = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise'
}

BATCH_SIZE = 128

# dataLoader = DataLoader(OurDataset(split='all', dataset='fer2013'), shuffle=True, batch_size=32)
dataLoader = DataLoader(OurDatasetTuning(section='architecture', split='train'), shuffle=True, batch_size=BATCH_SIZE)
batch = next(iter(dataLoader))
images = batch['image']
labels = batch['label']
    
rows = 8
cols = 16
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
axes = axes.flatten()
    
for i in range(BATCH_SIZE):
    image = images[i].numpy().squeeze()
    label_id = labels[i].item()
        
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(label_map.get(label_id, f"Unknown ({label_id})"), fontsize=10)
    axes[i].axis('off')

print("Mean of batch images:", np.mean(images.numpy()))
        
plt.tight_layout()
plt.savefig("Data/Plots/batchplot_processed_data.png")
