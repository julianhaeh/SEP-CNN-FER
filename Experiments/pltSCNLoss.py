""""
This is a script for fitting a Self Correcting Network (SCN), implemented in clsSCNWrapperOfVGG13.py, on our dataset.
It produces a lost history plot and confusion matrix for each optimizer tested. 
The SCN comes from the paper "Suppressing Uncertainties for Large-Scale Facial Expression Recognition" by Wang et al. 2020.
In their paper they provided an python implementation, which this script as well as the SCN wrapper are heavily based on. 
We adjusted some details, since we encountered crashes. We also tried to update the relabeling logic to the one described in the paper, where
it would only update the low importance samples instead of considering all.
The SCN loss work by adding a self-attention mechanism to to the architecture, which can weight the models importance, thus weightening samples whose correct
label is unclear, either because of wrong labeling or ambiguous facial expression. It adds a rank regularization loss to the standard cross-entropy loss, 
which forces the model to give higher attention weights to "easy" samples, thus only eliminating the uncertain samples during training. In addition to that, 
the SCN relabels samples when it is really certain about its prediction, which can lead to the model cleaning up the wrong labeling.  
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDatasetSCN import OurDatasetSCN
from ModelArchitectures.clsSCNWrapperOfVGG13 import SCN_VGG_Wrapper

# Log for the best performing hyperparameters Beta=0.8592, M1=0.0784, M2=0.4786, RelabelEp=14
# Hyperparameters used in paper: Beta=0.7, M1=0.15, M2=0.2, RelabelEp= --

# --- PARAMETERS ---
EPOCHS = 20
BATCH_SIZE = 1024  # SCN requires large batches for stable ranking
BETA = 0.86         # Ratio of "High Importance" samples (0.7 = 70%)
MARGIN_1 = 0.15    # Rank Regularization Margin
MARGIN_2 = 0.48   # Relabeling Margin
RELABEL_EPOCHS = 14 # Start relabeling after these many epochs
RELABELING = False # Enable/Disable relabeling

# --- DATA LOADERS ---
# drop_last=True is vital for SCN because the 'beta' calculation depends on batch_size
trainDataLoader = DataLoader(OurDatasetSCN(split='train'), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valDataLoader = DataLoader(OurDatasetSCN(split='test'), batch_size=BATCH_SIZE, shuffle=False)

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

# Global weights 
# class_weights = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # These weights can be used for testing unweighted CE

USE_SCHEDULER = False # Determines wether to use the Cosine Annealing Scheduler

# --- OPTIMIZERS ---
optimizer_configs = [
    ("Adam", optim.Adam, {"lr":  0.0001}),
    ("AdamW", optim.AdamW, {"lr": 0.001, "weight_decay": 1e-2}),
    ("SGD Momentum", optim.SGD, {"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4})
]


# --- HELPER FUNCTIONS ---
def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            # SCN Wrapper returns (alpha, logits), we need logits [1]
            outputs = model(inputs)
            _, preds = torch.max(outputs[1], 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_labels), torch.cat(all_preds)

def compute_confusion_matrix_torch(true_labels, pred_labels, num_classes=6):
    indices = true_labels * num_classes + pred_labels
    cm = torch.bincount(indices, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes).float()

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# --- TRAINING LOOP (With Integrated Logic) ---

def train_evaluate_pipeline(model, optimizer, scheduler, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []
    
    # Initialize Criterion with Class Weights (This handles the WCE part)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(trainDataLoader, desc=f"Ep {epoch+1}/{epochs}", leave=False)
        
        # Updated loop to unpack dictionary from OurDataset
        for batch in train_loop:
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            indexes = batch['index'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: returns (attention_weights, weighted_logits)
            attention_weights, outputs = model(imgs)
            
            batch_sz = imgs.size(0)
            RR_loss = 0.0

            # --- SCN RANK REGULARIZATION LOGIC ---
            # We wrap this in a warm-up check. If epoch < 5, RR_loss remains 0.0
            
            # Based on your provided snippet:
            tops = int(batch_sz * BETA)
                
            # Squeeze ensures shape is [Batch], not [Batch, 1]
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
                
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
                
            # Ensure High Group > Low Group + Margin
            # diff = low_mean - high_mean + MARGIN_1
            diff = low_mean - high_mean + MARGIN_1

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0

            # This is the relabeling logic of SNC, note that in the implementation by Wang et al. not only the low importance samples are relabeled, 
            # but this logic was described in the paper. Thus, we tried it out since the implemented version did not work out for us.
            if epoch >= RELABEL_EPOCHS and RELABELING:
                with torch.no_grad():
                    sm = torch.softmax(outputs, dim=1)
                    Pmax, predicted_labels = torch.max(sm, 1)
                    Pgt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()
                    
                    true_or_false = Pmax - Pgt > MARGIN_2
                    # is_low_importance = torch.zeros(batch_sz, dtype=torch.bool, device=device)
                    # is_low_importance[down_idx] = True
                    
                    true_or_false = true_or_false # & is_low_importance
                    update_idx = true_or_false.nonzero().reshape(-1)
                    
                    label_idx = indexes[update_idx] # get samples' index in train_loader
                    relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
                    trainDataLoader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
            
            # --- TOTAL LOSS ---
            # criterion(outputs, targets) is Logit-Weighted CE because 'outputs' are weighted logits
            loss = criterion(outputs, targets) + RR_loss
            
            loss.backward()
            optimizer.step()
            if scheduler is not None: 
                scheduler.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}", rr_loss=f"{RR_loss:.4f}")

        loss_history.append(running_loss / len(trainDataLoader))
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        correct = torch.sum(y_true == y_pred).item()
        total = y_true.size(0)
        accuracy = correct / total * 100
        print(f"   [Test Accuracy for Epoch {epoch}: {accuracy:.2f}% ]")
        


    return loss_history

def run_experiments():
    print("Starting SCN Experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)

    for opt_name, opt_class, opt_kwargs in optimizer_configs:
        print(f"\n=== Testing Optimizer: {opt_name} ===")
        
        # 1. INIT ARCHITECTURE
        base_model = CustomVGG13Reduced()
        model = SCN_VGG_Wrapper(base_model)

        # 2. INIT OPTIMIZER & SCHEDULER
        params = list(model.parameters())  
        optimizer = opt_class(params, **opt_kwargs)
        if USE_SCHEDULER:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=EPOCHS, 
                eta_min=0
            )
        else: scheduler = None
        
        # 3. TRAIN (Loss logic is now inside the function)
        history = train_evaluate_pipeline(model, optimizer, scheduler, epochs=EPOCHS)

        # 4. PLOTTING & METRICS
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history) + 1), history, marker='o', label="SCN Loss")
        plt.title(f"Training Loss: {opt_name} + SCN")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        loss_filename = f"Experiments/Plots/Loss_{opt_name}_SCN.png"
        plt.savefig(loss_filename)
        plt.close()
        print(f"   [Saved Loss Plot to {loss_filename}]")

        # Confusion Matrix
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)

        # Print final test accuracy
        correct = torch.sum(y_true == y_pred).item()
        total = y_true.size(0)
        accuracy = correct / total * 100
        print(f"   [Test Accuracy for {opt_name}: {accuracy:.2f}% ]")
        
        cm_filename = f"Experiments/Plots/ConfMatrix_{opt_name}_SCN.png"
        plot_confusion_matrix(cm_tensor.numpy(), CLASS_NAMES, 
                              f"Confusion Matrix: {opt_name} , SCN", 
                              cm_filename)
        print(f"   [Saved CM to {cm_filename}]")

if __name__ == "__main__":
    run_experiments()