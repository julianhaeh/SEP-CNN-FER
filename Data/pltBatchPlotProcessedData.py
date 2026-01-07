import matplotlib.pyplot as plt
from datasets import load_from_disk
import random
from clsOurDataset import OurDataset
from torch.utils.data import DataLoader
import numpy as np

label_map = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise'
}


dataLoader = DataLoader(OurDataset(split='all', dataset='fer2013'), shuffle=True, batch_size=32)
batch = next(iter(dataLoader))
images = batch['image']
labels = batch['label']
    
rows = 4
cols = 8
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
fig.suptitle(f"Batchplot of the processed data", fontsize=16)
    
axes = axes.flatten()
    
for i in range(32):
    image = images[i].numpy().squeeze()
    label_id = labels[i].item()
        
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(label_map.get(label_id, f"Unknown ({label_id})"), fontsize=10)
    axes[i].axis('off')
        
plt.tight_layout()
plt.savefig("Data/Plots/batchplot_processed_data.png")
