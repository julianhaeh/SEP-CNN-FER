"""
This script runs experiments to evaluate different loss functions and optimizers on the CustomVGG13Reduced architecture.
It generates combined training loss and accuracy plots per Loss function (comparing all optimizers).
It appends final test results to 'loss_optimization_log.txt'.
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
import torch.nn.init as init

# --- CUSTOM IMPORTS ---
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDatasetTuning import OurDatasetTuning

# --- PARAMETERS ---
EPOCHS = 45
BATCH_SIZE = 32

# Load Datasets
trainDataLoader = DataLoader(OurDatasetTuning(split='train'), batch_size=BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(OurDatasetTuning(split='valid'), batch_size=BATCH_SIZE, shuffle=False)

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

# Global weights 
class_weights = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

USE_SCHEDULER = True # Determines whether to use the Cosine Annealing Scheduler
LOG_FILE = "loss_optimization_log.txt"

# --- CONFIGURATIONS ---

loss_configs = [
    ("Weighted_CE", nn.CrossEntropyLoss(weight=class_weights)),
    ("Unweighted_CE", nn.CrossEntropyLoss())
]

optimizer_configs = [
    ("Adam", optim.Adam, {"lr": 0.0001}),
    ("AdamW", optim.AdamW, {"lr": 0.0001, "weight_decay": 1e-4}),
    ("SGD", optim.SGD, {"lr": 0.014, "momentum": 0.9, "weight_decay": 2.2e-4})
]

# --- HELPER FUNCTIONS ---

# Weight Intit for SGD, this stops gradient explosion or vanishing gradient
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # "fan_out" preserves magnitude in the backward pass
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01) # VGG paper specific for FC layers
        init.constant_(m.bias, 0)

def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            output = model(inputs)
            _, preds = torch.max(output, 1)
            
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

def plot_metric_comparison(history_dict, metric_name, loss_name, filename):
    """
    Plots a specific metric (Loss or Accuracy) for all optimizers on one chart.
    history_dict: { 'Adam': [values...], 'SGD': [values...] }
    """
    plt.figure(figsize=(10, 6))
    
    for opt_name, history in history_dict.items():
        plt.plot(range(1, len(history) + 1), history, marker='', linewidth=2, label=opt_name)
    
    plt.title(f"{metric_name} Comparison: {loss_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"   [Saved {metric_name} Comparison Plot to {filename}]")

# --- TRAINING LOOP ---

def train_evaluate_pipeline(model, criterion, optimizer, scheduler, epochs=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []
    accuracy_history = []
    
    if isinstance(criterion, nn.Module):
        criterion.to(device)
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
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
           
        if scheduler is not None:
            scheduler.step()
        loss_history.append(running_loss / len(trainDataLoader))
        
        # Validation
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        correct = torch.sum(y_true == y_pred).item()
        total = y_true.size(0)
        accuracy = correct / total * 100
        
        print(f"   [Test Accuracy for Epoch {epoch+1}: {accuracy:.2f}% ]")
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history

def run_experiments():
    print("Starting Experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)

    # LOOP 1: Iterate over LOSS FUNCTIONS first
    for loss_name, loss_fn in loss_configs:
        print(f"\n=========================================")
        print(f" Evaluating Loss Function: {loss_name}")
        print(f"=========================================")
        
        # Dictionaries to store history for ALL optimizers for this specific loss
        combined_loss_history = {}
        combined_accuracy_history = {}

        # LOOP 2: Iterate over OPTIMIZERS
        for opt_name, opt_class, opt_kwargs in optimizer_configs:
            print(f"\n   >>> Training with Optimizer: {opt_name}")
            
            # 1. INIT ARCHITECTURE
            model = CustomVGG13Reduced()
            model.apply(weights_init)   

            # 2. INIT OPTIMIZER & SCHEDULER
            params = list(model.parameters())  
            optimizer = opt_class(params, **opt_kwargs)
            
            if USE_SCHEDULER:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=EPOCHS
                )
            else:
                scheduler = None
            
            # 3. TRAIN
            l_hist, a_hist = train_evaluate_pipeline(model, loss_fn, optimizer, scheduler, epochs=EPOCHS)
            
            # Store history for combined plotting
            combined_loss_history[opt_name] = l_hist
            combined_accuracy_history[opt_name] = a_hist

            # 4. CONFUSION MATRIX (Per Run)
            y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
            cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)
            
            final_acc = a_hist[-1]
            print(f"   [Final Test Accuracy for {opt_name}: {final_acc:.2f}% ]")
            
            # --- NEW: LOG RESULTS TO FILE ---
            with open(LOG_FILE, "a") as f:
                f.write(f"Loss: {loss_name} | Optimizer: {opt_name} | Final Acc: {final_acc:.2f}%\n")
            print(f"   [Logged result to {LOG_FILE}]")
            
            cm_filename = f"Experiments/Plots/ConfMatrix_{loss_name}_{opt_name}.png"
            plot_confusion_matrix(cm_tensor.numpy(), CLASS_NAMES, 
                                  f"CM: {loss_name} + {opt_name}", 
                                  cm_filename)
        
        # --- END OF OPTIMIZER LOOP ---
        # Generate the combined plots for this specific Loss Function
        
        print(f"\n   >> Generating Combined Plots for {loss_name}...")
        
        loss_plot_path = f"Experiments/Plots/Comparison_Loss_{loss_name}.png"
        plot_metric_comparison(combined_loss_history, "Training Loss", loss_name, loss_plot_path)

        acc_plot_path = f"Experiments/Plots/Comparison_Accuracy_{loss_name}.png"
        plot_metric_comparison(combined_accuracy_history, "Validation Accuracy", loss_name, acc_plot_path)

if __name__ == "__main__":
    run_experiments()