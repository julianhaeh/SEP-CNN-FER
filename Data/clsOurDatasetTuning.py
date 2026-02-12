"""
This file defines the OurDatasetTuning class, which downloads and preprocesses the AffectNet and FER2013 datasets from HuggingFace. 
This dataset defined the tuning splits for our work, there are two different splits accesed by the section argument. The split 
for tuning the optimizers, the optimizers hyperparameters, the loss function and the data agumentation is accessed by section='training',
this corresponds to the training section of our paper. It used the first 15% of training data as validation set. The other split is accessed  
by section='architecture', this split is used for hyperparamter tuning of the architectures and uses the last 15% of the training data 
as validation set. 
"""
import numpy as np
from datasets import load_dataset, concatenate_datasets, ClassLabel
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Target resolution for all images
TARGET_SIZE = (64, 64)
SHUFFLE_SEED = 42

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

class OurDatasetTuning(Dataset):
    """
    The dataset class for this project. Downloads the dataset from HuggingFace. 
    """

    def __init__(self, section, dataset = 'all', split = 'train', custom_transform=None):
        """
        dataset = 'all', 'affectnet' or 'fer2013' 
        split = 'train', 'test', 'valid' or 'all'
        """

        self.split = split

        self.TrainTransform = v2.Compose([
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            v2.RandomHorizontalFlip(p=0.5),            
        ])

        self.custom_transform = custom_transform

        self.tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
        if(dataset == 'all'):

            affectnetDs = load_dataset("Mauregato/affectnet_short", split='train')
            fer2013Ds = load_dataset("AutumnQiu/fer2013", split='train+valid')


            
            print("Processing AffectNet " + split + "...")
            affectnetDs = affectnetDs.map(
                process_and_filter, 
                batched=True, 
                batch_size=1000, 
                remove_columns=["label", "image"]
            )

            print("Processing FER2013 " + split + "...")
            fer2013Ds = fer2013Ds.map(
                process_fer, 
                batched=True, 
                batch_size=1000, 
                remove_columns=["label", "image"]
            )

            new_label_feature = ClassLabel(names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])
            affectnetDs = affectnetDs.cast_column('label', new_label_feature)
            fer2013Ds = fer2013Ds.cast_column('label', new_label_feature)

            # The fixed shuffle seed here is really important, since this ensures we have two different validation sets
            # for the training and the architecture part. The idx determines where to split the data into train in valid.
            # This index is determined by the section argument and changes for Training Validation Set or Architecture Validation Set.
        
            combined_ds = concatenate_datasets([affectnetDs, fer2013Ds])
            shuffled_ds = combined_ds.shuffle(SHUFFLE_SEED)
            labels = np.array(shuffled_ds['label'])
            images = np.array(shuffled_ds['image'])

            # We have two 
            if section == 'training':

                idx = int(0.15 * len(shuffled_ds)) # Take the first 15% of the shuffled dataset as validation for the training part

                if split == 'train':

                    self.label = labels[idx:]
                    self.image = images[idx:]
                    self.original_label = labels[idx:]

                elif split == 'valid':

                    self.label = labels[:idx]
                    self.image = images[:idx]
                    self.original_label = labels[:idx]

            elif section == 'architecture':

                idx = int(0.85 * len(shuffled_ds)) # Take the last 15% of the shuffled dataset as validation for the architecture part
                
                if split == 'train':

                    self.label = labels[:idx]
                    self.image = images[:idx]
                    self.original_label = labels[:idx]

                elif split == 'valid':

                    self.label = labels[idx:]
                    self.image = images[idx:]
                    self.original_label = labels[idx:]
            

    def __getitem__(self, idx):

  
        img = self.image[idx]
        label = self.label[idx]
        
        img_tensor = self.tensor(img)

        if self.split == 'train' and self.custom_transform is None:
            img_tensor = self.TrainTransform(img_tensor)

        elif self.split == 'train' and self.custom_transform is not None:
            img_tensor = self.custom_transform(img_tensor)

        img_tensor = self.normalize(img_tensor)
        
        return {"image" : img_tensor, "label" : label}
    
    def __len__(self):
 
        return len(self.label)