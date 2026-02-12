import torch.nn.init as init
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
import random
from torch.optim import lr_scheduler
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDatasetTuning import OurDatasetTuning
from torchvision.transforms import v2

# --- PARAMETERS ---
EPOCHS = 45
BATCH_SIZE = 32
NUM_RUNS = 3  # Number of models to train per transformation
SEEDS = [42, 101, 2024]  # Fixed seeds for reproducibility across transformations

VISUALIZE_FIRST_EPOCH = False
AUG_LOG_FILE = "Experiments/Plots/data_augmentation_stats.txt"

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

valDataLoader = DataLoader(OurDatasetTuning(section='training', split='valid'), batch_size=BATCH_SIZE, shuffle=False)
      
"""
transforms_list = [
    ("No Augmentation", v2.Identity()),
    ("Horizontal Flip", v2.RandomHorizontalFlip(p=0.5)),
    ("Translation", v2.RandomAffine(degrees=0, translate=(0.1, 0.1))),
    ("Rotation", v2.RandomRotation(degrees=15)),
    ("Random Erasing", v2.RandomErasing(p=0.25)),
    ("Combined", v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.RandomRotation(degrees=15),]))    
]
"""

transforms_list = [
    ("Flip, Affine, Rotation", v2.Compose([v2.RandomHorizontalFlip(p=0.5), 
                               v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                               v2.RandomRotation(degrees=15)])),

    ("Flip, Affine",           v2.Compose([v2.RandomHorizontalFlip(p=0.5), 
                               v2.RandomAffine(degrees=0, translate=(0.1, 0.1))])),

    ("Flip, Rotation",         v2.Compose([v2.RandomHorizontalFlip(p=0.5),        
                               v2.RandomRotation(degrees=15)])),

    ("Rotation, Affine",       v2.Compose([v2.RandomRotation(degrees=15),
                               v2.RandomAffine(degrees=0, translate=(0.1, 0.1))]))
]

# Global weights 
class_weights = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])
criterion = nn.CrossEntropyLoss(weight=class_weights)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # "fan_out" preserves magnitude in the backward pass
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01) # VGG paper specific for FC layers
        init.constant_(m.bias, 0)

def set_seed(seed):
    """Ensures reproducibility across runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = torch.tensor([], dtype=torch.long)
    all_labels = torch.tensor([], dtype=torch.long)
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            output = model(inputs)
            _, preds = torch.max(output, 1)
            
            all_preds = torch.cat((all_preds, preds.cpu()))
            all_labels = torch.cat((all_labels, labels.cpu()))
            
    return all_labels, all_preds

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
    Plots a specific metric for all RUNS on one chart.
    history_dict: { 'Run 1': [values...], 'Run 2': [values...] }
    """
    plt.figure(figsize=(10, 6))
    
    for run_name, history in history_dict.items():
        plt.plot(range(1, len(history) + 1), history, marker='', label=run_name, linewidth=1.5, alpha=0.8)
    
    plt.title(f"{metric_name} - {loss_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def train_evaluate_pipeline(model, optimizer, scheduler, dataloader, epochs=20, run_id=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []
    accuracy_history = []
    criterion.to(device)
    criterion.weight.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(dataloader, desc=f"Run {run_id+1} | Ep {epoch+1}/{epochs}", leave=False)
        
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

        scheduler.step()
        loss_history.append(running_loss / len(dataloader))
        
        # Validation
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        correct = torch.sum(y_true == y_pred).item()
        total = y_true.size(0)
        accuracy = correct / total * 100
        
        print(f"   [Run {run_id+1} - Ep {epoch+1}] Acc: {accuracy:.2f}%")
            
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history

def run_experiments():
    print(f"Starting Experiments ({NUM_RUNS} runs per transform)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)
    
    # Initialize Log File
    with open(AUG_LOG_FILE, "a") as f:
        f.write(f"Data Augmentation Experiments (Avg of {NUM_RUNS} runs)\n")
        f.write("====================================================\n")

    for transform_name, transform in transforms_list:
        print(f"\n=========================================")
        print(f" Evaluating Transform: {transform_name}")
        print(f"=========================================")
        
        # Store histories for plotting all 5 runs
        run_loss_histories = {}
        run_acc_histories = {}
        
        # Store final scores for statistical logging
        final_scores = []

        for i in range(NUM_RUNS):
            current_seed = SEEDS[i]
            set_seed(current_seed) 
            
            # Re-init Model & Optimizer
            model = CustomVGG13Reduced()
            model.apply(weights_init)  # Initialize weights
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            
            # Re-init Loader (ensures shuffle is reset based on seed)
            trainDataLoader = DataLoader(OurDatasetTuning(split='train', custom_transform=transform), 
                                         batch_size=BATCH_SIZE, shuffle=True)

            print(f" > Starting Run {i+1}/{NUM_RUNS} (Seed {current_seed})...")
            
            l_hist, a_hist = train_evaluate_pipeline(
                model, optimizer, scheduler, trainDataLoader, epochs=EPOCHS, run_id=i
            )
            
            run_loss_histories[f'Run {i+1}'] = l_hist
            run_acc_histories[f'Run {i+1}'] = a_hist
            
            # METRIC: Average of last 3 epochs of THIS run
            # This represents the performance of this specific model
            run_score = sum(a_hist[NUM_RUNS * -1:]) / NUM_RUNS
            final_scores.append(run_score)

        # --- AGGREGATE RESULTS ---
        avg_acc = np.mean(final_scores)
        std_acc = np.std(final_scores)

        # Log to file
        with open(AUG_LOG_FILE, "a") as f:
            f.write(f"{transform_name}: {avg_acc:.2f}% (+/- {std_acc:.2f})\n")
        
        print(f"\n >>> {transform_name} RESULT: {avg_acc:.2f}% ± {std_acc:.2f}")

        # --- PLOTTING (Show all 5 runs) ---
        print(f"     Generating Combined Plots...")
        
        loss_plot_path = f"Experiments/Plots/Stability_Loss_{transform_name}.png"
        plot_metric_comparison(run_loss_histories, "Training Loss", transform_name, loss_plot_path)

        acc_plot_path = f"Experiments/Plots/Stability_Accuracy_{transform_name}.png"
        plot_metric_comparison(run_acc_histories, "Validation Accuracy", transform_name, acc_plot_path)

        # --- CONFUSION MATRIX (Use the LAST model trained, just for reference) ---
        y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)
        cm_filename = f"Experiments/Plots/ConfMatrix_{transform_name}.png"
        plot_confusion_matrix(cm_tensor.numpy(), CLASS_NAMES, f"{transform_name} (Run {NUM_RUNS})", cm_filename)

if __name__ == "__main__":
    run_experiments()