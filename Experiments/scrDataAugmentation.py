"""

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
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDataset import OurDataset
from torchvision.transforms import v2

# --- PARAMETERS ---
EPOCHS = 1
BATCH_SIZE = 1024
VISUALIZE_FIRST_EPOCH = True

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

valDataLoader = DataLoader(OurDataset(split='test'), batch_size=BATCH_SIZE, shuffle=False)

transforms_list = [
    # ("No Augmentation", v2.Identity()),
    ("Horizontal Flip", v2.RandomHorizontalFlip(p=0.5)),
    ("Translation", v2.RandomAffine(degrees=0, translate=(0.1, 0.1))),
    ("Rotation", v2.RandomRotation(degrees=15)),
    ("Random Erasing", v2.RandomErasing(p=0.25))
]

# Global weights 
class_weights = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

USE_SCHEDULER = False # Determines wether to use the Cosine Annealing Scheduler

criterion = nn.CrossEntropyLoss(weight=class_weights)

def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Model returns logits directly in your architecture
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

# --- PLOTTING HELPERS (NEW) ---

def plot_metric_comparison(history_dict, metric_name, loss_name, filename):
    """
    Plots a specific metric (Loss or Accuracy) for all optimizers on one chart.
    history_dict: { 'Adam': [values...], 'SGD': [values...] }
    """
    plt.figure(figsize=(10, 6))
    
    for opt_name, history in history_dict.items():
        plt.plot(range(1, len(history) + 1), history, marker='o', label=opt_name)
    
    plt.title(f"{metric_name} Comparison: {loss_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"   [Saved {metric_name} Comparison Plot to {filename}]")

# --- TRAINING LOOP ---

def train_evaluate_pipeline(model, optimizer, scheduler, dataloader, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []
    accuracy_history = []
    criterion.to(device)
    criterion.weight.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(dataloader, desc=f"Ep {epoch+1}/{epochs}", leave=False)

        if epoch == 0 and VISUALIZE_FIRST_EPOCH:

            rows = 4
            cols = 8
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
            batch = next(iter(dataloader))
            images = batch['image'][:32]
            labels = batch['label'][:32]

            axes = axes.flatten()

            for i in range(32):
                image = images[i].numpy().squeeze()
                label_id = labels[i].item()
                    
                axes[i].imshow(image, cmap='gray')
                axes[i].set_title(EMOTION_DICT.get(label_id, f"Unknown ({label_id})"), fontsize=10)
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

        
        for batch in train_loop:
            
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
                
            optimizer.zero_grad()
                
            # Forward pass
            outputs = model(imgs)
                
            loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        loss_history.append(running_loss / len(dataloader))
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        correct = torch.sum(y_true == y_pred).item()
        total = y_true.size(0)
        accuracy = correct / total * 100
        print(f"   [Test Accuracy for Epoch {epoch}: {accuracy:.2f}% ]")
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history

def run_experiments():
    print("Starting Experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)

    # LOOP 1: Iterate over LOSS FUNCTIONS first
    for transform_name, transform in transforms_list:
        print(f"\n=========================================")
        print(f" Evaluating Transform: {transform_name}")
        print(f"=========================================")
        
        # Dictionaries to store history for ALL optimizers for this specific loss
        combined_loss_history = {}
        combined_accuracy_history = {}

        model = CustomVGG13Reduced()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        if USE_SCHEDULER:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer)
       
        trainDataLoader = DataLoader(OurDataset(split='train', custom_transform=transform), batch_size=BATCH_SIZE, shuffle=True)

        l_hist, a_hist = train_evaluate_pipeline(model, optimizer, scheduler=None, dataloader=trainDataLoader, epochs=EPOCHS)
        # 4. CONFUSION MATRIX (Still generated per specific run)
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)
            
        # Print final accuracy
        final_acc = a_hist[-1]
        print(f"   [Final Test Accuracy for Transformation {transform_name}: {final_acc:.2f}% ]")
            
        cm_filename = f"Experiments/Plots/ConfMatrix_{transform_name}.png"
        plot_confusion_matrix(cm_tensor.numpy(), CLASS_NAMES, 
                                f"Transform: {transform_name}", 
                                cm_filename)
        
        # --- END OF OPTIMIZER LOOP ---
        # Now generate the combined plots for this specific Loss Function
        
        print(f"\n   >> Generating Combined Plots for {transform_name}...")
        
        # Plot 1: Loss Comparison
        loss_plot_path = f"Experiments/Plots/Comparison_Loss_{transform_name}.png"
        plot_metric_comparison(combined_loss_history, "Training Loss", transform_name, loss_plot_path)

        # Plot 2: Accuracy Comparison
        acc_plot_path = f"Experiments/Plots/Comparison_Accuracy_{transform_name}.png"
        plot_metric_comparison(combined_accuracy_history, "Validation Accuracy", transform_name, acc_plot_path)

if __name__ == "__main__":
    run_experiments()