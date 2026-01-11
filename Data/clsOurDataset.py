
from datasets import load_dataset, concatenate_datasets, ClassLabel
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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

class OurDataset(Dataset):
    """" 
    The dataset class for this project. Downloads the dataset from HuggingFace. 
    """

    def __init__(self, dataset = 'all', split = 'train'):
        """"
        dataset = 'all', 'affectnet' or 'fer2013' 
        split = 'train', 'test' or 'all'
        """

        if(dataset == 'all'):

            if split == 'train':
                affectnetDs = load_dataset("Mauregato/affectnet_short", split='train')
                fer2013Ds = load_dataset("AutumnQiu/fer2013", split='train+valid')
            elif split == 'test':
                affectnetDs = load_dataset("Mauregato/affectnet_short", split='val')
                fer2013Ds = load_dataset("AutumnQiu/fer2013", split='test')
            elif split == 'all':
                affectnetDs = load_dataset("Mauregato/affectnet_short", split='train+val')
                fer2013Ds = load_dataset("AutumnQiu/fer2013", split='train+valid+test')
            else: 
                raise ValueError("Split must be 'train', 'test' or 'all'.")

            
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
        
            combined_ds = concatenate_datasets([affectnetDs, fer2013Ds])
            shuffled_ds = combined_ds.shuffle(SHUFFLE_SEED)
            
            self.data = shuffled_ds
        
        elif(dataset == 'affectnet'):

            if split == 'train':
                affectnetDs = load_dataset("Mauregato/affectnet_short", split='train')
            elif split == 'test':
                affectnetDs = load_dataset("Mauregato/affectnet_short", split='val')
            elif split == 'all':
                affectnetDs = load_dataset("Mauregato/affectnet_short", split='train+val')
            else: 
                raise ValueError("Split must be 'train', 'test' or 'all'.")
            
            print("Processing AffectNet " + split + "...")
            affectnetDs = affectnetDs.map(
                process_and_filter, 
                batched=True, 
                batch_size=1000, 
                remove_columns=["label", "image"]
            )

            new_label_feature = ClassLabel(names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])
            affectnetDs = affectnetDs.cast_column('label', new_label_feature)
        
            shuffled_ds = affectnetDs.shuffle(SHUFFLE_SEED)
            
            self.data = shuffled_ds

        elif(dataset == 'fer2013'):

            if split == 'train':
                fer2013Ds = load_dataset("AutumnQiu/fer2013", split='train+valid')
            elif split == 'test':
                fer2013Ds = load_dataset("AutumnQiu/fer2013", split='test')
            elif split == 'all':
                fer2013Ds = load_dataset("AutumnQiu/fer2013", split='train+valid+test')
            else: 
                raise ValueError("Split must be 'train', 'test' or 'all'.")
            
            print("Processing FER2013 " + split + "...")
            fer2013Ds = fer2013Ds.map(
                process_fer, 
                batched=True, 
                batch_size=1000, 
                remove_columns=["label", "image"]
            )

            new_label_feature = ClassLabel(names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])
            fer2013Ds = fer2013Ds.cast_column('label', new_label_feature)
        
            shuffled_ds = fer2013Ds.shuffle(SHUFFLE_SEED)
            
            self.data = shuffled_ds
        else: 
            raise ValueError("Dataset must be 'all', 'affectnet' or 'fer2013'.")
            

    def __getitem__(self, idx):

        example = self.data[idx]
        
        img = example['image']
        label = example['label']
        
        # Convert PIL image to tensor [C, H, W] with values in [0, 1]
        img_tensor = transforms.ToTensor()(img)
        
        return {"image" : img_tensor, "label" : label}
    
    def __len__(self):
 
        return len(self.data)