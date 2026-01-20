"""
This script runs experiments to evaluate different loss functions and optimizers on the CustomVGG13Reduced architecture.
It generates training loss plots and confusion matrices for each combination tested.
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
from torch.optim import lr_scheduler

# --- CUSTOM IMPORTS ---
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDataset import OurDataset
# from ModelArchitectures.clsIslandLoss import IslandLoss # Example import

# --- PARAMETERS ---
EPOCHS = 20
BATCH_SIZE = 1024

trainDataLoader = DataLoader(OurDataset(split='train'), batch_size=BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(OurDataset(split='test'), batch_size=BATCH_SIZE, shuffle=False)

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

# Global weights 
class_weights = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

USE_SCHEDULER = False # Determines wether to use the Cosine Annealing Scheduler

# --- CONFIGURATIONS ---

loss_configs = [
    ("Weighted CE", nn.CrossEntropyLoss(weight=class_weights)),
#    ("Unweighted CE", nn.CrossEntropyLoss())
]

optimizer_configs = [
    ("Adam", optim.Adam, {"lr":  0.0001}),
#    ("AdamW", optim.AdamW, {"lr": 0.001, "weight_decay": 1e-2}),
#    ("SGD Momentum", optim.SGD, {"lr": 0.001, "momentum": 0.9, "weight_decay": 1e-4})
]

# --- HELPER FUNCTIONS ---

def calculate_loss(criterion, outputs, labels):
    """
    Generic loss calculator.
    Handles standard losses (Logits only) and Feature-based losses (Logits + Features).
    """
    device = labels.device
    
    # Unpack outputs (Our model returns features, logits)
    features, logits = outputs

    # CASE A: Standard PyTorch Loss (e.g., CrossEntropy)
    if isinstance(criterion, nn.CrossEntropyLoss):
        return criterion(logits, labels)
    
    # CASE B: Custom Losses (e.g., Island Loss)
    # If the criterion is a custom module, we assume it handles its own logic
    elif isinstance(criterion, nn.Module):
         # Heuristic: If it has 'centers' (like Island/Center loss), it needs features
        if hasattr(criterion, 'centers'): 
             # Calculate the auxiliary loss on features
             aux_loss = criterion(features, labels)
             # Add standard Classification Loss (Weighted)
             ce_loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
             return ce_loss + aux_loss
        else:
             # Default fallback
             return criterion(logits, labels)
             
    return criterion(logits, labels)

def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Model returns (features, logits) -> we want logits [1]
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

# --- TRAINING LOOP ---

def train_evaluate_pipeline(model, criterion, optimizer, scheduler, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []
    
    # Ensure criterion is on device (important for custom losses with parameters)
    if isinstance(criterion, nn.Module):
        criterion.to(device)

    # Ensure class weights inside CrossEntropy are on device
    if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(trainDataLoader, desc=f"Ep {epoch+1}/{epochs}", leave=False)
        
        for batch in train_loop:
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass -> (features, logits)
            outputs = model(imgs)
            
            loss = calculate_loss(criterion, outputs, targets)
            
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        loss_history.append(running_loss / len(trainDataLoader))

    return loss_history

def run_experiments():
    print("Starting Experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)

    for opt_name, opt_class, opt_kwargs in optimizer_configs:
        print(f"\n=== Testing Optimizer: {opt_name} ===")
        
        for loss_name, loss_fn in loss_configs:
            print(f"   >> Loss Function: {loss_name}")
            
            # 1. INIT ARCHITECTURE
            model = CustomVGG13Reduced()
            
            # 2. INIT OPTIMIZER & SCHEDULER
            params = list(model.parameters())  
            optimizer = opt_class(params, **opt_kwargs)
            if USE_SCHEDULER:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=EPOCHS, 
                    eta_min=0
                )
            else:
                scheduler = None
            
            # 3. TRAIN
            history = train_evaluate_pipeline(model, loss_fn, optimizer, scheduler, epochs=EPOCHS)

            # 4. PLOTTING
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(history) + 1), history, marker='o', label=loss_name)
            plt.title(f"Training Loss: {opt_name} + {loss_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            loss_filename = f"Experiments/Plots/Loss_{opt_name}_{loss_name}.png"
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
            
            cm_filename = f"Experiments/Plots/ConfMatrix_{opt_name}_{loss_name}.png"
            plot_confusion_matrix(cm_tensor.numpy(), CLASS_NAMES, 
                                  f"Confusion Matrix: {opt_name} , {loss_name}", 
                                  cm_filename)
            print(f"   [Saved CM to {cm_filename}]")

if __name__ == "__main__":
    run_experiments()