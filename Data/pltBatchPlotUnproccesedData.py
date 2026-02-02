"""
This script generates and saves a batch plot of raw images from Hugging Face datasets.
"""
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
import math

# --- DEBUG VARIABLES ---
# Options: "AFFECTNET" or "FER2013"
DATASET_NAME = "AFFECTNET" 
BATCH_SIZE = 16

# Mapping for emotion labels

if DATASET_NAME == "FER2013":
    label_map = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }
else: label_map = {
        0: "anger",
        1: "surprise",
        2: "contempt",
        3: "happy",
        4: "neutral",
        5: "fear",
        6: "sad",
        7: "disgust"
    }

# --- DATA LOADING ---
if DATASET_NAME == "AFFECTNET":
    ds = load_dataset("Mauregato/affectnet_short", split='train+val')
else:
    ds = load_dataset("AutumnQiu/fer2013", split='train+valid+test')

# Shuffle and select a batch
ds_batch = ds.shuffle().select(range(BATCH_SIZE))

# --- PLOTTING ---
# Dynamically calculate grid dimensions based on BATCH_SIZE
cols = 4
rows = math.ceil(BATCH_SIZE / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

for i in range(BATCH_SIZE):
    example = ds_batch[i]
    image = example['image']
    label_id = example['label']
    
    # Convert PIL image to numpy array if it's not already
    img_array = np.array(image)
    
    # Handle grayscale vs RGB for imshow
    if len(img_array.shape) == 2:
        axes[i].imshow(img_array, cmap='gray')
    else:
        axes[i].imshow(img_array)
        
    axes[i].set_title(label_map.get(label_id, f"ID: {label_id}"), fontsize=10)
    axes[i].axis('off')

# Hide any empty subplots if BATCH_SIZE is not a multiple of cols
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(f"Data/Plots/batchplot_{DATASET_NAME.lower()}_data.png")
print(f"Plot saved as Data/Plots/batchplot_{DATASET_NAME.lower()}_data.png")