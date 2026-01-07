import matplotlib.pyplot as plt
from clsOurDataset import OurDataset
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

DATASET = 'fer2013'

label_map = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise'
}

dataLoader = DataLoader(OurDataset(split='all', dataset=DATASET), batch_size=256)

histogram_counts = np.zeros(6, dtype=int)

for batch in tqdm(dataLoader, desc="Scanning dataset"):
    labels = batch['label'].numpy()
    
    batch_counts = np.bincount(labels, minlength=6)
    
    histogram_counts += batch_counts

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(label_map.values(), histogram_counts, color='skyblue', edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_title(f"Class Distribution (Total: {histogram_counts.sum()})", fontsize=16)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_xlabel("Emotion Class", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Data/Plots/class_distribution_" + DATASET + "_histogram.png")