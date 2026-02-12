"""
This script implements Bayesian Optimization using Optuna to tune hyperparameters for training a SCN. 
The hyperparameters and their evaluation metrics are logged to Experiments/Plots/scn_optimization_history.txt.
"""
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import optuna

# --- USER IMPORTS ---
# Assuming these are available in your directory structure
from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced
from Data.clsOurDatasetSCN import OurDatasetTuning
from ModelArchitectures.clsSCNWrapperOfVGG13 import SCN_VGG_Wrapper

# --- CONSTANTS ---
EPOCHS = 30
BATCH_SIZE = 64
LOG_FILE = "Experiments/Plots/scn_optimization_history.txt"
RELABEL_EPOCH = 15

# --- DEBUG CONSTANTS ---
RELABELING_ENABLED = True # Set to True to enable relabeling
PAPER_RELABELING = True  # Set to True to use the relabeling logic as described in the paper. This will only relabel low-importance samples, in the orignal repo its both.

CLASS_WEIGHTS = torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
# CLASS_WEIGHTS = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])
PRETRAINED_WEIGHTS_PATH = "Experiments/Models/CustomVGG13_Downsized_Acc_72.51_Model.pth"


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

def evaluate_model(model, loader, device, criterion):
    """Calculates Loss and Accuracy on the Test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # SCN Wrapper returns (alpha, raw_logits, logits). We only need logits [2] for evaluation
            outputs = model(inputs)
            logits = outputs[2]
            
            # Loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def objective(trial):
    # --- 1. HYPERPARAMETER SEARCH SPACE ---
    # Optuna will suggest values from these ranges using TPE (Bayesian optimization)
    BETA = trial.suggest_float("beta", 0.2, 0.9, step=0.02)
    MARGIN_1 = trial.suggest_float("margin_1", 0.1, 0.8, step=0.02) 
    MARGIN_2 = trial.suggest_float("margin_2", 0.1, 0.8, step=0.02)
    GAMMA = trial.suggest_float("gamma", 0.1, 0.9, step=0.02)
    
    # --- 2. SETUP (Fresh for every trial) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMPORTANT: We MUST re-initialize the dataset/loader every trial.
    # The SCN logic modifies labels in-place: trainDataLoader.dataset.data[idx]['label'] = ...
    # If we reuse the loader, Trial 2 starts with Trial 1's modified labels.
    train_dataset = OurDatasetTuning(section='architecture', split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Validation loader can be reused (read-only), but defining here for safety
    val_loader = DataLoader(OurDatasetTuning(section='architecture', split='test'), batch_size=BATCH_SIZE, shuffle=False)
    # Init Model
    base_model = DownsizedCustomVGG13Reduced()
    
    base_model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location='cpu'))
    model = SCN_VGG_Wrapper(base_model)
    model.to(device)
    
    # Init Optimizer 
    # optimizer = optim.SGD(model.parameters(), lr=0.014, momentum=0.9, weight_decay=2.2e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # --- 3. TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        
        # We only use tqdm for the outer trial loop to avoid cluttering output
        # or use a simplified print
        
        train_loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)

        for batch in train_loop:
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            indexes = batch['index'].to(device)
            
            optimizer.zero_grad()
            
            # Forward: (alpha, logits)
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
            
            # --- RELABELING LOGIC ---
            if epoch >= RELABEL_EPOCH:
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
                            train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy()
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
                            train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy()
                

            # --- LOSS CALCULATION ---
            loss = (1 - GAMMA) * criterion(outputs, targets) + GAMMA * RR_loss
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(
            trial=trial.number, 
            epoch=f"{epoch+1}/{EPOCHS}", 
            loss=f"{loss.item():.4f}"
        )
            
        scheduler.step()

        
    # --- 4. FINAL EVALUATION ---
    final_test_loss, final_accuracy = evaluate_model(model, val_loader, device, criterion)

    print(f"Trial {trial.number} completed: Test Loss={final_test_loss:.4f}, Accuracy={final_accuracy:.2f}%")
    
    # --- 5. LOGGING ---
    with open(LOG_FILE, "a") as f:
        f.write(f"Trial {trial.number}: Loss={final_test_loss:.4f}, Acc={final_accuracy:.2f}% | "
                f"Beta={BETA:.4f}, M1={MARGIN_1:.4f}, M2={MARGIN_2:.4f}, Gamma={GAMMA:.4f}\n")
    
    return final_accuracy

if __name__ == "__main__":
    # Create the log file with header if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("SCN Hyperparameter Optimization History\n")
            f.write("=======================================\n")

    print("Starting Bayesian Optimization with Optuna...")
    
    study = optuna.create_study(direction="maximize")
    
    # n_trials=20 (You can increase this if you have time)
    study.optimize(objective, n_trials=60)

    print("\noptimization finished!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Test Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")