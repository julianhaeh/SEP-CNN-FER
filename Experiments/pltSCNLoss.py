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
from torch.utils.data import DataLoader
from torch.nn import init
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDatasetSCN import OurDatasetSCN
from ModelArchitectures.clsSCNWrapperOfVGG13 import SCN_VGG_Wrapper

# --- EXPERIMENT CONSTANTS ---
EPOCHS = 30
BATCH_SIZE = 64
USE_SCHEDULER = True
RELABEL_EPOCH = 10

# --- DEBUG CONSTANTS ---
RELABELING_ENABLED = True  # Set to True to enable relabeling
PAPER_RELABELING = True  # Set to True to use the relabeling logic as described in the paper
DEBUG_BATCH_PLOT = True  # Set to True to enable batch plotting
PLOT_EPOCHS = [1, 10, 25, 30]  # Epochs to plot (1-indexed)
NUM_BATCHES_TO_PLOT = 1  # Number of batches to plot per epoch
PRINT_WHEN_RELABELD = True # Print info when a sample is relabeled
PRINT_GROUP_MEANS = True # Print mean attention weights for high and low importance groups once per epoch

# Class mappings
EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
CLASS_NAMES = [val for key, val in sorted(EMOTION_DICT.items())]

# Global Class Weights (for Weighted CE)
CLASS_WEIGHTS_TENSOR = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

# --- CONFIGURATIONS ---
# 1. Hyperparameter Configs
HYPERPARAM_CONFIGS = {
    "Tuned": {
        "BETA": 0.7200,
        "MARGIN_1": 0.2800,
        "MARGIN_2": 0.4600,
        "GAMMA": 0.2200
    },
     "Original": {
        "BETA": 0.7,
        "MARGIN_1": 0.15,
        "MARGIN_2": 0.2,
        "GAMMA": 0.5
    }
}

# 2. Loss Configs (Loss_name, use_weighted, path_pretrained_model)
LOSS_CONFIGS = [
    ("Unweighted CE", False, "Experiments/Models/VGG13_Original_Unweighted_CE_Acc_72.20.pth"),
    ("Weighted CE", True, "Experiments/Models/VGG13_Original_Unweighted_CE_Acc_72.20.pth")
]

# --- HELPER FUNCTIONS ---

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)

def plot_attention_weights_vs_loss(all_attention_weights, y_true, y_logits, experiment_title):

    # Convert to numpy
    if isinstance(all_attention_weights, torch.Tensor):
        att_weights_np = all_attention_weights.numpy()
    else:
        att_weights_np = np.array(all_attention_weights)

    ce = nn.CrossEntropyLoss(reduction='none')
    loss_per_sample = ce(torch.tensor(y_logits, dtype=torch.float32), torch.tensor(y_true, dtype=torch.long)).numpy()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(att_weights_np, loss_per_sample, alpha=0.6)
    plt.title(f"Attention Weight vs CE-loss")
    plt.xlabel("Attention Weight (α)", fontsize=12)
    plt.ylabel("CE-loss", fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    
    filename = os.path.join("Experiments/Plots", f"Attention_Weights_{experiment_title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention weight distribution plot saved: {filename}")


def plot_relabeling_comparison(dataset, relabeled_mask, config_name, save_dir="Experiments/Plots"):
    """
    Creates a 2x4 plot showing relabeling effects:
    - Top row: 4 relabeled samples with original and new labels
    - Bottom row: 4 non-relabeled samples with original labels
    
    Args:
        dataset: The training dataset (OurDatasetSCN) with both label and original_label attributes
        relabeled_mask: Boolean tensor/array where True indicates the sample was relabeled
        config_name: Name of the configuration for the filename
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert mask to numpy if it's a tensor
    if isinstance(relabeled_mask, torch.Tensor):
        relabeled_mask = relabeled_mask.numpy()
    
    # Get indices of relabeled and non-relabeled samples
    relabeled_indices = np.where(relabeled_mask)[0]
    non_relabeled_indices = np.where(~relabeled_mask)[0]
    
    # Check if we have enough samples
    if len(relabeled_indices) < 4:
        print(f"Warning: Only {len(relabeled_indices)} relabeled samples found. Need at least 4.")
        return
    if len(non_relabeled_indices) < 4:
        print(f"Warning: Only {len(non_relabeled_indices)} non-relabeled samples found. Need at least 4.")
        return
    
    # Randomly select 4 relabeled and 4 non-relabeled samples
    np.random.seed(42)  # For reproducibility
    selected_relabeled = np.random.choice(relabeled_indices, size=4, replace=False)
    selected_non_relabeled = np.random.choice(non_relabeled_indices, size=4, replace=False)
    
    # Create 2x4 subplot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Top row: Relabeled samples
    for col, idx in enumerate(selected_relabeled):
        ax = axes[0, col]
        
        # Get the sample
        sample = dataset[idx]
        img = sample['image']
        current_label = sample['label']
        original_label = sample['original_label']
        
        # Convert tensor to numpy for display
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
            # Handle different channel configurations
            if img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)  # Remove channel dim for grayscale
                ax.imshow(img_np, cmap='gray')
            elif img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
                # Denormalize if necessary (assuming standard ImageNet normalization)
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                ax.imshow(img_np)
            else:
                ax.imshow(img_np.squeeze(0), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        
        # Get label names
        original_name = CLASS_NAMES[original_label] if original_label < len(CLASS_NAMES) else f"Unknown({original_label})"
        current_name = CLASS_NAMES[current_label] if current_label < len(CLASS_NAMES) else f"Unknown({current_label})"
        
        # Set title with both labels
        ax.set_title(f"Original: {original_name}\nRelabeled: {current_name}", 
                     fontsize=22, fontweight='bold', color='red')
        
        # Add red border to indicate relabeling
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
        
        ax.axis('off')
    
    # Bottom row: Non-relabeled samples
    for col, idx in enumerate(selected_non_relabeled):
        ax = axes[1, col]
        
        # Get the sample
        sample = dataset[idx]
        img = sample['image']
        original_label = sample['label']
        
        # Convert tensor to numpy for display
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
            # Handle different channel configurations
            if img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)  # Remove channel dim for grayscale
                ax.imshow(img_np, cmap='gray')
            elif img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
                # Denormalize if necessary (assuming standard ImageNet normalization)
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                ax.imshow(img_np)
            else:
                ax.imshow(img_np.squeeze(0), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        
        # Get label name
        original_name = CLASS_NAMES[original_label] if original_label < len(CLASS_NAMES) else f"Unknown({original_label})"
        
        # Set title with original label only
        ax.set_title(f"Original: {original_name}", 
                     fontsize=22, fontweight='bold', color='green')
        
        # Add green border to indicate no relabeling
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    filename = os.path.join(save_dir, f"Relabeling_Comparison_{config_name.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Relabeling comparison plot saved: {filename}")


def plot_batch_debug(imgs, labels, attention_weights, outputs, epoch, batch_idx, config_name, 
                     high_idx=None, low_idx=None, save_dir="Experiments/DebugPlots"):
    """
    Creates a debug plot for a batch showing images, labels, and attention weights.
    Also compares loss with and without attention weighting.
    
    Args:
        imgs: Tensor of images [batch, channels, height, width]
        labels: Tensor of labels [batch]
        attention_weights: Tensor of attention weights [batch, 1]
        outputs: Tensor of weighted logits (attention_weights * original_logits) [batch, num_classes]
        epoch: Current epoch (1-indexed)
        batch_idx: Current batch index
        config_name: Name of the configuration
        high_idx: Indices of high-importance samples (optional)
        low_idx: Indices of low-importance samples (optional)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = imgs.size(0)
    
    # --- Compute loss comparison ---
    # Recover original logits: original_logits = outputs / attention_weights
    # outputs shape: [batch, num_classes], attention_weights shape: [batch, 1]
    with torch.no_grad():
        original_logits = outputs / attention_weights  # Broadcasting handles the division
        
        # Compute losses
        criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
        
        # Loss WITH attention weighting (current SCN behavior)
        loss_with_attention = criterion(outputs, labels)
        
        # Loss WITHOUT attention weighting (standard CE on original logits)
        loss_without_attention = criterion(original_logits, labels)
        
        # Aggregate losses
        total_loss_with_att = loss_with_attention.mean().item()
        total_loss_without_att = loss_without_attention.mean().item()
        
        # Per-sample losses for display
        per_sample_loss_with = loss_with_attention.detach().cpu().numpy()
        per_sample_loss_without = loss_without_attention.detach().cpu().numpy()

        # Per-sample ligts for display 
        per_sample_logits_with = outputs.detach().cpu().numpy()
        per_sample_logits_without = original_logits.detach().cpu().numpy()

    
    # Determine grid size (try to make it roughly square)
    ncols = min(8, batch_size)
    nrows = (batch_size + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.5))
    
    # Ensure axes is 2D for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Convert tensors to numpy (detach to remove from computation graph)
    imgs_np = imgs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    attention_np = attention_weights.squeeze().detach().cpu().numpy()
    
    # Get min/max attention for color scaling
    att_min = attention_np.min()
    att_max = attention_np.max()
    att_mean = attention_np.mean()
    att_std = attention_np.std()
    
    # Convert indices to sets for fast lookup (detach in case they require grad)
    high_set = set(high_idx.detach().cpu().numpy().tolist()) if high_idx is not None else set()
    low_set = set(low_idx.detach().cpu().numpy().tolist()) if low_idx is not None else set()
    
    for idx in range(batch_size):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Get image (handle single channel)
        img = imgs_np[idx]
        if img.shape[0] == 1:
            img = img.squeeze(0)  # Remove channel dim for grayscale
            ax.imshow(img, cmap='gray')
        else:
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            ax.imshow(img)
        
        # Get label and attention weight
        label = labels_np[idx]
        att_weight = attention_np[idx] if attention_np.ndim > 0 else attention_np.item()
        true_label = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Unknown({label})"
        
        # Get per-sample losses
        loss_with = per_sample_loss_with[idx]
        loss_without = per_sample_loss_without[idx]

        prediction = np.argmax(per_sample_logits_with[idx])
        pred_label = CLASS_NAMES[prediction] if prediction < len(CLASS_NAMES) else f"Unknown({prediction})"

        # Get per-sample logits
        logits_with = per_sample_logits_with[idx, prediction]
        logits_without = per_sample_logits_without[idx, prediction]
        
        # Determine importance group
        if idx in high_set:
            group = "HIGH"
            border_color = 'green'
        elif idx in low_set:
            group = "LOW"
            border_color = 'red'
        else:
            group = "?"
            border_color = 'gray'
        
        # Set title with label, attention weight, and loss comparison
        title = f"True={true_label}\nPred={pred_label}\nα={att_weight:.3f} [{group}]\nL_scn={loss_with:.2f} L_orig={loss_without:.2f}\nLog_SCN={logits_with:.2f} Log_Orig={logits_without:.2f}"
        ax.set_title(title, fontsize=8, color=border_color, fontweight='bold')
        
        # Add colored border based on importance group
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(batch_size, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    # Add overall title with statistics and loss comparison
    fig.suptitle(
        f"Epoch {epoch} | Batch {batch_idx} | Config: {config_name}\n"
        f"Attention Stats: min={att_min:.3f}, max={att_max:.3f}, mean={att_mean:.3f}, std={att_std:.3f}\n"
        f"Batch Loss: WITH attention={total_loss_with_att:.4f} | WITHOUT attention={total_loss_without_att:.4f} | Δ={total_loss_with_att - total_loss_without_att:.4f}",
        fontsize=11, fontweight='bold'
    )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', linewidth=3, label='High Importance'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Low Importance')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    filename = os.path.join(save_dir, f"batch_debug_{config_name}_epoch{epoch:02d}_batch{batch_idx:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   [DEBUG] Saved batch plot: {filename}")



def get_all_predictions_torch(model, loader, device):
    model.eval()
    all_preds = torch.tensor([], dtype=torch.long)
    all_labels = torch.tensor([], dtype=torch.long)
    all_attention_weights = torch.tensor([])
    all_raw_logits = torch.tensor([])
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            attention_weights, raw_logits, logits = model(inputs)
            # SCN Wrapper returns (alpha, logits), we need logits [1]
            _, preds = torch.max(logits, 1)
            all_preds = torch.cat((all_preds, preds.cpu()))
            all_labels = torch.cat((all_labels, labels.cpu()))
            all_attention_weights = torch.cat((all_attention_weights, attention_weights.cpu()))
            all_raw_logits = torch.cat((all_raw_logits, raw_logits.cpu()))
    return all_raw_logits, all_attention_weights, all_labels, all_preds

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


def plot_history(history, title, filename, ylabel="Value"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


# --- TRAINING PIPELINE ---

def train_evaluate_pipeline(hp_config, use_weighted_loss, path_pretrained_model, config_name=""):
    """
    Runs a full training session for one specific configuration.
    Returns loss_history, accuracy_history, and final predictions.
    """
    # 1. Unpack Hyperparameters for this run
    BETA = hp_config["BETA"]
    MARGIN_1 = hp_config["MARGIN_1"]
    MARGIN_2 = hp_config["MARGIN_2"]
    GAMMA = hp_config.get("GAMMA")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Re-Initialize DataLoaders (CRITICAL for SCN to reset labels)
    trainDataLoader = DataLoader(OurDatasetSCN(split='train'), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valDataLoader = DataLoader(OurDatasetSCN(split='test'), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model & Weights
    base_model = CustomVGG13Reduced()
    base_model.load_state_dict(torch.load(path_pretrained_model, map_location='cpu'))
    model = SCN_VGG_Wrapper(base_model)
    model.to(device)
    
    # 4. Initialize Optimizer (Fixed SGD)
    # optimizer = optim.SGD(model.parameters(), lr=0.014, momentum=0.9, weight_decay=2.2e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
        
    # 5. Define Loss
    if use_weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS_TENSOR.to(device))
    else:
        # Default unweighted
        criterion = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []
    
    # --- EPOCH LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(trainDataLoader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        relabels_this_epoch = 0
        groups_printed_this_epoch = 0
        
        # Track how many batches we've plotted this epoch
        batches_plotted_this_epoch = 0
        
        for batch_idx, batch in enumerate(train_loop):
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            indexes = batch['index'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: returns (attention_weights, raw_logits, outputs)
            attention_weights, raw_logits, outputs = model(imgs)
            batch_sz = imgs.size(0)
            
            # --- SCN RANK REGULARIZATION ---
            RR_loss = 0.0
            
            tops = int(batch_sz * BETA)
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
                
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
                
            diff = low_mean - high_mean + MARGIN_1
            if diff > 0:
                RR_loss = diff

            if PRINT_GROUP_MEANS and groups_printed_this_epoch == 0:
                print(f"   [Epoch {epoch+1}] High group mean α: {high_mean.item():.4f}, Low group mean α: {low_mean.item():.4f}, Difference + MARGIN_1: {diff.item():.4f}")
                groups_printed_this_epoch += 1
            
            # --- DEBUG: Batch Plotting ---
            if DEBUG_BATCH_PLOT and (epoch + 1) in PLOT_EPOCHS and batches_plotted_this_epoch < NUM_BATCHES_TO_PLOT:
                plot_batch_debug(
                    imgs=imgs,
                    labels=targets,
                    attention_weights=attention_weights,
                    outputs=outputs,
                    epoch=epoch + 1,
                    batch_idx=batch_idx,
                    config_name=config_name,
                    high_idx=top_idx,
                    low_idx=down_idx
                )
                batches_plotted_this_epoch += 1
            
            # --- SCN RELABELING LOGIC ---
            if RELABELING_ENABLED and epoch >= RELABEL_EPOCH:
                with torch.no_grad():
                    if not PAPER_RELABELING:
                        sm = torch.softmax(outputs, dim=1)
                        Pmax, predicted_labels = torch.max(sm, 1)
                        Pgt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()
                        
                        # Relabeling condition
                        true_or_false = Pmax - Pgt > MARGIN_2
                        update_idx = true_or_false.nonzero().reshape(-1)
                        
                        label_idx = indexes[update_idx] 
                        relabels = predicted_labels[update_idx]
                        
                        # Update dataset labels in place
                        label_idx_np = label_idx.cpu().numpy()
                        trainDataLoader.dataset.label[label_idx_np] = relabels.cpu().numpy()
                        relabels_this_epoch += len(label_idx)
                    else:
                        sm = torch.softmax(outputs, dim=1)
                        Pmax, predicted_labels = torch.max(sm, 1)
                        Pgt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()

                        low_importance_mask = torch.zeros(batch_sz, dtype=torch.bool, device=device)
                        low_importance_mask[down_idx] = True
                        
                        # Relabeling condition
                        true_or_false = (Pmax - Pgt > MARGIN_2) & low_importance_mask
                        update_idx = true_or_false.nonzero().reshape(-1)
                        
                        label_idx = indexes[update_idx] 
                        relabels = predicted_labels[update_idx]
                        
                        # Update dataset labels in place
                        label_idx_np = label_idx.cpu().numpy()
                        trainDataLoader.dataset.label[label_idx_np] = relabels.cpu().numpy()
                        relabels_this_epoch += len(label_idx)

            
            # Total Loss
            loss = (1 - GAMMA) * criterion(outputs, targets) +  GAMMA * RR_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}", rr_loss=f"{RR_loss * GAMMA:.4f}")

            
        if PRINT_WHEN_RELABELD and epoch >= RELABEL_EPOCH:
                print(f"   [Epoch {epoch+1} Relabeled {relabels_this_epoch} samples.")

        if scheduler:
            scheduler.step()
            
        # Record training metrics
        loss_history.append(running_loss / len(trainDataLoader))
        
        # Validation Loop for Accuracy History
        all_raw_logits, all_attention_weights, y_true, y_pred = get_all_predictions_torch(model, valDataLoader, device)
        correct = torch.sum(y_true == y_pred).item()
        total = y_true.size(0)
        epoch_acc = correct / total * 100
        acc_history.append(epoch_acc)
        
        print(f"   [Epoch {epoch+1} Test Acc: {epoch_acc:.2f}%]")
        print(f"  [Epoch {epoch+1}] Training Loss: {loss_history[-1]:.4f}")

    # Get final predictions for Confusion Matrix
    all_raw_logits, all_attention_weights, y_true_final, y_pred_final = get_all_predictions_torch(model, valDataLoader, device)
    
    return loss_history, acc_history, y_true_final, y_pred_final, all_attention_weights, all_raw_logits, trainDataLoader


# --- MAIN EXPERIMENT LOOP ---

def run_experiments():
    print("Starting SCN Experiments...")
    print(f"Debug batch plotting: {DEBUG_BATCH_PLOT}")
    if DEBUG_BATCH_PLOT:
        print(f"  Plotting epochs: {PLOT_EPOCHS}")
        print(f"  Batches per epoch: {NUM_BATCHES_TO_PLOT}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loop over configurations (Original vs Tuned)
    for config_name, config_params in HYPERPARAM_CONFIGS.items():
        
        # Loop over Loss types (Weighted vs Unweighted)
        for loss_name, use_weighted, path_pretrained_model in LOSS_CONFIGS:
            
            # Construct the combined title part
            experiment_title = f"{config_name} {loss_name}"
            print(f"\n==============================================")
            print(f"Running Experiment: {experiment_title}")
            print(f"Params: {config_params}")
            print(f"==============================================")
            
            # Run Pipeline (pass config_name for debug plots)
            loss_hist, acc_hist, y_true, y_pred, all_attention_weights, all_raw_logits, trainDataLoader = train_evaluate_pipeline(
                config_params, 
                use_weighted,
                path_pretrained_model,
                config_name=f"{config_name}_{loss_name.replace(' ', '')}"
            )
            
            # 1. Plot Loss History
            plot_title_loss = f"Loss History {experiment_title}"
            filename_loss = f"Experiments/Plots/Loss_{config_name}_{loss_name.replace(' ', '')}.png"
            plot_history(loss_hist, plot_title_loss, filename_loss, ylabel="Loss")
            
            # 2. Plot Test Accuracy History
            plot_title_acc = f"Test Accuracy {experiment_title}"
            filename_acc = f"Experiments/Plots/Accuracy_{config_name}_{loss_name.replace(' ', '')}.png"
            plot_history(acc_hist, plot_title_acc, filename_acc, ylabel="Accuracy (%)")
            
            # 3. Plot Confusion Matrix
            cm_tensor = compute_confusion_matrix_torch(y_true, y_pred, num_classes=6)
            plot_title_cm = f"Confusion Matrix {experiment_title}"
            filename_cm = f"Experiments/Plots/CM_{config_name}_{loss_name.replace(' ', '')}.png"
            
            plot_confusion_matrix(
                cm_tensor.numpy(), 
                CLASS_NAMES, 
                plot_title_cm, 
                filename_cm
            )

            # 4. Plot Relabeling Comparison
            # Compute relabeled indices by comparing current labels to original labels
            dataset = trainDataLoader.dataset
            relabeled_mask = dataset.label != dataset.original_label
            num_relabeled = relabeled_mask.sum().item() if isinstance(relabeled_mask, torch.Tensor) else relabeled_mask.sum()
            print(f"\nTotal number of relabeled samples: {num_relabeled}")
            
            if num_relabeled >= 2:
                plot_relabeling_comparison(
                    dataset,
                    relabeled_mask,
                    experiment_title
                )
            else:
                print("Not enough relabeled samples to create comparison plot.")
            
            print(f"Completed {experiment_title}. Plots saved.")

            # 5. Plot Attention Weight vs CE-loss without attention
            plot_attention_weights_vs_loss(all_attention_weights, y_true, all_raw_logits, experiment_title)


if __name__ == "__main__":
    run_experiments()