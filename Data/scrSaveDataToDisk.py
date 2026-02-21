""""
Script to save the processed data to the disk. 
Data is already processed in order to save space, saving the data is important in case it gets deleted online.
"""

import os
import numpy as np
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset as HFDataset
from PIL import Image

SHUFFLE_SEED = 42

def process_and_filter(batch):
    new_images = []
    new_labels = []
    
    for img, label in zip(batch["image"], batch["label"]):
        if label in [2, 4]:
            continue
            
        if label == 0: new_lbl = 0   # Anger
        elif label == 7: new_lbl = 1 # Disgust
        elif label == 5: new_lbl = 2 # Fear
        elif label == 3: new_lbl = 3 # Happy
        elif label == 6: new_lbl = 4 # Sad
        elif label == 1: new_lbl = 5 # Surprise
        else: raise ValueError(f"Unexpected label {label} in AffectNet dataset.")
        
        img_resized = img.resize((64, 64), resample=Image.Resampling.LANCZOS)
        new_images.append(img_resized.convert("L"))
        new_labels.append(new_lbl)
    
    return {"image": new_images, "label": new_labels}

def process_fer(batch):
    new_images = []
    new_labels = []
    for img, label in zip(batch["image"], batch["label"]):
        if label == 6: continue
        
        img_resized = img.resize((64, 64), resample=Image.Resampling.LANCZOS)
        new_images.append(img_resized.convert("L"))
        new_labels.append(label)
    return {"image": new_images, "label": new_labels}

def process_dataset(ds, process_fn, num_cores):
    ds = ds.map(
        process_fn, 
        batched=True, 
        batch_size=1000, 
        num_proc=num_cores, 
        remove_columns=["label", "image"]
    )
    new_label_feature = ClassLabel(names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])
    ds = ds.cast_column('label', new_label_feature)
    return ds

if __name__ == "__main__":

    SAVE_DATA_PATH = ""
    if SAVE_DATA_PATH == "": 
        raise ValueError("Specify a path to save the processed datasets.")

    num_cores = os.cpu_count()
    print(f"Processing with {num_cores} CPU cores...")

    # ===== 1. Process and save the TEST split (used by OurDataset with split='test') =====
    print("\n--- Processing TEST split ---")
    affectnet_test = load_dataset("Mauregato/affectnet_short", split="val")
    fer2013_test = load_dataset("AutumnQiu/fer2013", split="test")

    affectnet_test = process_dataset(affectnet_test, process_and_filter, num_cores)
    fer2013_test = process_dataset(fer2013_test, process_fer, num_cores)

    test_ds = concatenate_datasets([affectnet_test, fer2013_test])
    test_ds = test_ds.shuffle(SHUFFLE_SEED)

    test_path = os.path.join(SAVE_DATA_PATH, "test")
    os.makedirs(test_path, exist_ok=True)
    test_ds.save_to_disk(test_path)
    print(f"Saved test split ({len(test_ds)} samples) to {test_path}")

    # ===== 2. Process the TRAIN split (used by OurDataset with split='train' and OurDatasetTuning) =====
    print("\n--- Processing TRAIN split ---")
    affectnet_train = load_dataset("Mauregato/affectnet_short", split="train")
    fer2013_train = load_dataset("AutumnQiu/fer2013", split="train+valid")

    affectnet_train = process_dataset(affectnet_train, process_and_filter, num_cores)
    fer2013_train = process_dataset(fer2013_train, process_fer, num_cores)

    train_ds = concatenate_datasets([affectnet_train, fer2013_train])
    train_ds = train_ds.shuffle(SHUFFLE_SEED)

    # Save full train split (used by OurDataset with split='train')
    train_path = os.path.join(SAVE_DATA_PATH, "train")
    os.makedirs(train_path, exist_ok=True)
    train_ds.save_to_disk(train_path)
    print(f"Saved full train split ({len(train_ds)} samples) to {train_path}")

    # ===== 3. Create and save the tuning validation splits (used by OurDatasetTuning) =====
    labels = np.array(train_ds['label'])
    n = len(train_ds)

    # Training section: first 15% = validation, rest = train
    training_val_idx = int(0.15 * n)

    training_valid_ds = train_ds.select(range(0, training_val_idx))
    training_train_ds = train_ds.select(range(training_val_idx, n))

    path = os.path.join(SAVE_DATA_PATH, "tuning", "training_section", "valid")
    os.makedirs(path, exist_ok=True)
    training_valid_ds.save_to_disk(path)
    print(f"Saved training section valid ({len(training_valid_ds)} samples) to {path}")

    path = os.path.join(SAVE_DATA_PATH, "tuning", "training_section", "train")
    os.makedirs(path, exist_ok=True)
    training_train_ds.save_to_disk(path)
    print(f"Saved training section train ({len(training_train_ds)} samples) to {path}")

    # Architecture section: last 15% = validation, rest = train
    architecture_val_idx = int(0.85 * n)

    architecture_train_ds = train_ds.select(range(0, architecture_val_idx))
    architecture_valid_ds = train_ds.select(range(architecture_val_idx, n))

    path = os.path.join(SAVE_DATA_PATH, "tuning", "architecture_section", "train")
    os.makedirs(path, exist_ok=True)
    architecture_train_ds.save_to_disk(path)
    print(f"Saved architecture section train ({len(architecture_train_ds)} samples) to {path}")

    path = os.path.join(SAVE_DATA_PATH, "tuning", "architecture_section", "valid")
    os.makedirs(path, exist_ok=True)
    architecture_valid_ds.save_to_disk(path)
    print(f"Saved architecture section valid ({len(architecture_valid_ds)} samples) to {path}")

    print("\nDone! Saved directory structure:")
    print(f"  {SAVE_DATA_PATH}/")
    print(f"    train/                              (full train, {len(train_ds)} samples)")
    print(f"    test/                               (test, {len(test_ds)} samples)")
    print(f"    tuning/")
    print(f"      training_section/")
    print(f"        train/                          ({len(training_train_ds)} samples)")
    print(f"        valid/                          ({len(training_valid_ds)} samples)")
    print(f"      architecture_section/")
    print(f"        train/                          ({len(architecture_train_ds)} samples)")
    print(f"        valid/                          ({len(architecture_valid_ds)} samples)")