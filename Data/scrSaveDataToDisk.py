""""
Script to save the processed data to the disk. 
Data is already processed in order to save space, saving the data is important in case it gets deleted online.
"""

import os
from datasets import load_dataset, ClassLabel
from PIL import Image

def process_and_filter(batch):
    new_images = []
    new_labels = []
    
    for img, label in zip(batch["image"], batch["label"]):
        # Filter AffectNet (Remove Contempt=2, Neutral=4)
        if label in [2, 4]:
            continue
            
        # Remaping AffectNet labels
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

if __name__ == "__main__":

    # Insert the desired save path here
    SAVE_DATA_PATH = ""
    if SAVE_DATA_PATH == "": raise ValueError("Specify a path to save the processed datasets.")

    print("Loading datasets...")
    affectnetDs = load_dataset("Mauregato/affectnet_short", split="train+val")
    fer2013Ds = load_dataset("AutumnQiu/fer2013", split="train+valid+test")

    num_cores = os.cpu_count()
    print(f"Processing with {num_cores} CPU cores...")

    print("Processing AffectNet...")
    affectnetDs = affectnetDs.map(
        process_and_filter, 
        batched=True, 
        batch_size=1000, 
        num_proc=num_cores, 
        remove_columns=["label", "image"]
    )

    print("Processing FER2013...")
    fer2013Ds = fer2013Ds.map(
        process_fer, 
        batched=True, 
        batch_size=1000, 
        num_proc=num_cores,
        remove_columns=["label", "image"]
    )

    new_label_feature = ClassLabel(names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])
    affectnetDs = affectnetDs.cast_column('label', new_label_feature)
    fer2013Ds = fer2013Ds.cast_column('label', new_label_feature)

    print("Saving to disk...")
    save_path_affect = SAVE_DATA_PATH + "/affectnet_short_filtered"
    save_path_fer = SAVE_DATA_PATH + "/fer2013_filtered"
    
    os.makedirs(save_path_affect, exist_ok=True)
    os.makedirs(save_path_fer, exist_ok=True)

    affectnetDs.save_to_disk(save_path_affect)
    fer2013Ds.save_to_disk(save_path_fer)
    
    print("Done!")