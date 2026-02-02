import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import torch.nn.init as init

# --- CUSTOM IMPORTS ---
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDataset import OurDataset
from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced

# --- PARAMETERS ---
EPOCHS = 75
BATCH_SIZE = 32

# Load Datasets
trainDataLoader = DataLoader(OurDataset(split='train'), batch_size=BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(OurDataset(split='test'), batch_size=BATCH_SIZE, shuffle=False)

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

# Global weights 
CLASS_WEIGHTS = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

# --- CONFIGURATIONS ---
LOSS_CONFIGS = [
    ("Weighted_CE", True),
    ("Unweighted_CE", False)
]

USE_PRETRAINED = None # Set to None to train from scratch, otherwise will load weights from the path
USE_ORIGINAL_VGG13 = False  # If True, uses the original CustomVGG13Reduced architecture, if false uses the downsized one

# --- HELPER FUNCTIONS ---

def weights_init(m):
    """Weight init for SGD, this stops gradient explosion or vanishing gradient"""
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
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


def plot_metric_comparison(history_dict, metric_name, filename):
    """
    Plots a specific metric (Loss or Accuracy) for all configurations on one chart.
    history_dict: { 'Weighted_CE': [values...], 'Unweighted_CE': [values...] }
    """
    plt.figure(figsize=(10, 6))
    
    for config_name, history in history_dict.items():
        plt.plot(range(1, len(history) + 1), history, marker='', linewidth=2, label=config_name)
    
    plt.title(f"{metric_name} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"   [Saved {metric_name} Comparison Plot to {filename}]")


# --- TRAINING LOOP ---

def train_evaluate_pipeline(model, criterion, optimizer, scheduler, epochs=55):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []
    accuracy_history = []
    
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
        
        print(f"   [Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%]")
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history


def run_experiments():
    print("Starting Experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)

    # Dictionaries to store history for both loss configurations
    combined_loss_history = {}
    combined_accuracy_history = {}

    # Loop over loss configurations
    for loss_name, use_weighted in LOSS_CONFIGS:
        print(f"\n=========================================")
        print(f" Training with: {loss_name}")
        print(f"=========================================")
        
        # 1. INIT ARCHITECTURE
        if USE_ORIGINAL_VGG13:
            model = CustomVGG13Reduced() 
            model_name = "Original"
        else:
            model = DownsizedCustomVGG13Reduced()
            model_name = "Downsized"
        model.apply(weights_init) 

        print(f"   [Model Initialized: {model_name}]")

        if USE_PRETRAINED is not None:
            model.load_state_dict(torch.load(USE_PRETRAINED, map_location=device))
            print(f"   [Loaded Pretrained Weights from {USE_PRETRAINED}]") 

        # 2. INIT LOSS
        if use_weighted:
            criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        # 3. INIT OPTIMIZER & SCHEDULER (SGD with fixed hyperparameters)
        optimizer = optim.SGD(model.parameters(), lr=0.014, momentum=0.9, weight_decay=2.2e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # 4. TRAIN
        l_hist, a_hist = train_evaluate_pipeline(model, criterion, optimizer, scheduler, epochs=EPOCHS)
        
        # Store history for combined plotting
        combined_loss_history[loss_name] = l_hist
        combined_accuracy_history[loss_name] = a_hist

        # 5. CONFUSION MATRIX
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)
        
        final_acc = a_hist[-1]
        print(f"\n   [Final Test Accuracy for {loss_name}: {final_acc:.2f}%]")
        
        cm_filename = f"Experiments/Plots/ConfMatrix_{loss_name}.png"
        plot_confusion_matrix(cm_tensor.numpy(), CLASS_NAMES, 
                              f"Confusion Matrix: {loss_name}", 
                              cm_filename)
        
        torch.save(model.state_dict(), f"Experiments/Models/CustomVGG13_{model_name}_Acc_{final_acc:.2f}_Model.pth")

    # --- Generate combined comparison plots ---
    print(f"\n=========================================")
    print(f" Generating Comparison Plots...")
    print(f"=========================================")
    
    loss_plot_path = f"Experiments/Plots/VGG13_{model_name}_Comparison_Loss.png"
    plot_metric_comparison(combined_loss_history, "Training Loss", loss_plot_path)

    acc_plot_path = f"Experiments/Plots/VGG13_{model_name}_Comparison_Accuracy.png"
    plot_metric_comparison(combined_accuracy_history, "Test Accuracy", acc_plot_path)
    
    print("\nExperiments completed.")


if __name__ == "__main__":
    run_experiments()