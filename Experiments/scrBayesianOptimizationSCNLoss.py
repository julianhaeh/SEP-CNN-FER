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
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from Data.clsOurDatasetTuning import OurDatasetTuning
from ModelArchitectures.clsSCNWrapperOfVGG13 import SCN_VGG_Wrapper

# --- CONSTANTS ---
EPOCHS = 55
BATCH_SIZE = 32
LOG_FILE = "Experiments/Plots/scn_optimization_history.txt"
RELABELING = True

CLASS_WEIGHTS = torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
# CLASS_WEIGHTS = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])


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
            
            # SCN Wrapper returns (alpha, logits). We only need logits [1] for evaluation
            outputs = model(inputs)
            logits = outputs[1]
            
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
    beta = trial.suggest_float("beta", 0.4, 0.95, step=0.01)
    margin_1 = trial.suggest_float("margin_1", 0.02, 0.5, step=0.01) 
    margin_2 = trial.suggest_float("margin_2", 0.05, 0.5, step=0.01)
    relabel_epochs = trial.suggest_int("relabel_epochs", 5, 54)
    # relabeling = trial.suggest_categorical("relabeling", [True, False])
    
    # --- 2. SETUP (Fresh for every trial) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMPORTANT: We MUST re-initialize the dataset/loader every trial.
    # The SCN logic modifies labels in-place: trainDataLoader.dataset.data[idx]['label'] = ...
    # If we reuse the loader, Trial 2 starts with Trial 1's modified labels.
    train_dataset = OurDatasetTuning(split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Validation loader can be reused (read-only), but defining here for safety
    val_loader = DataLoader(OurDatasetTuning(split='valid'), batch_size=BATCH_SIZE, shuffle=False)
    # Init Model
    base_model = CustomVGG13Reduced()
    model = SCN_VGG_Wrapper(base_model).to(device)
    model.apply(weights_init)
    
    # Init Optimizer 
    optimizer = optim.SGD(model.parameters(), lr=0.014, momentum=0.9, weight_decay=2.2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

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
            attention_weights, outputs = model(imgs)
            batch_sz = imgs.size(0)
            
            # --- SCN RANK REGULARIZATION ---
            RR_loss = 0.0
            
            tops = int(batch_sz * beta)
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)
            
            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            
            diff = low_mean - high_mean + margin_1
            if diff > 0:
                RR_loss = diff
            
            # --- RELABELING LOGIC ---
            if epoch >= relabel_epochs:
                with torch.no_grad():
                    sm = torch.softmax(outputs, dim=1)
                    Pmax, predicted_labels = torch.max(sm, 1)
                    Pgt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()
                    
                    true_or_false = Pmax - Pgt > margin_2
                    
                    true_or_false = true_or_false
                    update_idx = true_or_false.nonzero().reshape(-1)
                    
                    label_idx = indexes[update_idx] # get samples' index in train_loader
                    relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
                    train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
                

            # --- LOSS CALCULATION ---
            loss = criterion(outputs, targets) + RR_loss
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
                f"Beta={beta:.4f}, M1={margin_1:.4f}, M2={margin_2:.4f}, RelabelEp={relabel_epochs}\n")
    
    return final_test_loss

if __name__ == "__main__":
    # Create the log file with header if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("SCN Hyperparameter Optimization History\n")
            f.write("=======================================\n")

    print("Starting Bayesian Optimization with Optuna...")
    
    # Create a study object and optimize the objective function.
    # direction="minimize" because we want the lowest Test Loss.
    study = optuna.create_study(direction="minimize")
    
    # n_trials=20 (You can increase this if you have time)
    study.optimize(objective, n_trials=60)

    print("\noptimization finished!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Test Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")