"""
This script generates and saves a histogram plot showing the class distribution of the specified dataset split. 
"""

import matplotlib.pyplot as plt
from clsOurDataset import OurDataset
from torch.utils.data import DataLoader
from clsOurDatasetTuning import OurDatasetTuning
import numpy as np
import os
from tqdm import tqdm

DATASET = 'all'
SPLIT = 'valid'

# Ensure the output directory exists
os.makedirs("Data/Plots", exist_ok=True)

label_map = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise'
}

dataLoader = DataLoader(OurDatasetTuning(split=SPLIT, dataset=DATASET), batch_size=256)

histogram_counts = np.zeros(6, dtype=int)

for batch in tqdm(dataLoader, desc="Scanning dataset"):
    labels = batch['label'].numpy()
    # minlength=6 ensures we get a count for every class 0-5
    batch_counts = np.bincount(labels, minlength=6) 
    histogram_counts += batch_counts

# --- NEW: Generate Labels with Percentages ---
total_samples = histogram_counts.sum()
x_labels_with_pct = []

for i in range(6):
    count = histogram_counts[i]
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    # Creates a label like: "Anger\n(15.4%)"
    label_text = f"{label_map[i]}\n({percentage:.1f}%)"
    x_labels_with_pct.append(label_text)
# ---------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Use the new 'x_labels_with_pct' list for the x-axis
bars = ax.bar(x_labels_with_pct, histogram_counts, color='skyblue', edgecolor='black')

# Annotate bars with the raw counts
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# ax.set_title(f"Class Distribution (Total: {total_samples})", fontsize=16)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_xlabel("Emotion Class", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

filename = f"Data/Plots/class_distribution_{DATASET}_{SPLIT}_histogram.png"
plt.savefig(filename)
print(f"Plot saved to {filename}")