import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import optuna  
from torch.optim import lr_scheduler
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from ModelArchitectures.clsCustomVGG13ReducedBatchnorm import CustomVGG13ReducedBatchnorm
from ModelArchitectures.clsOurCNNArchitecture import CNN_GAP_3Blocks
from Data.clsOurDatasetTuning import OurDatasetTuning
import torch.nn.init as init

# --- CONFIGURATION ---
MODEL = CNN_GAP_3Blocks # Set the model architecture to tune (e.g. CustomVGG13Reduced, CustomVGG13ReducedBatchnorm, CNN_GAP_3Blocks)
N_TRIALS = 70
EPOCHS_PER_TRIAL = 15
BATCH_SIZE = 32
SEED = 42

# Global weights 
class_weights = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            output = model(inputs)
            _, preds = torch.max(output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

def objective(trial):
    """
    The Optimization Function. Optuna will call this repeatedly with 
    different hyperparameters proposed by the Bayesian algorithm.
    """
    
    # 1. SAMPLE HYPERPARAMETERS
    # suggest_float with log=True allows efficient searching across magnitudes (e.g. 0.001 vs 0.0001)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True) 
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    # Optional: Tune Cosine Annealing settings?
    # t_max_multiplier = trial.suggest_int("t_max_mult", 1, 2) 

    # 2. SETUP DATA & MODEL
    set_seed(SEED) # Reset seed so every trial starts from same weights/batch order
    
    # Load Data (Re-init to ensure clean state)
    trainDataLoader = DataLoader(OurDatasetTuning(section='training', split='train'),
                                 batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valDataLoader = DataLoader(OurDatasetTuning(section='training', split='valid'), 
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MODEL().to(device)
    model.apply(weights_init)  # Initialize weights
    
    # Initialize Optimizer with TRIAL parameters
    # Note: We use SGD with Momentum as recommended in the paper
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    
    # Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 3. TRAINING LOOP
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for batch in trainDataLoader:
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()

            
            
            optimizer.step()
        
        scheduler.step()

        # 4. VALIDATION & PRUNING
        # We optimize for Validation Accuracy
        val_acc = get_accuracy(model, valDataLoader, device)
        
        # Report intermediate result to Optuna
        trial.report(val_acc, epoch)

        # Handle Pruning (Stop bad trials early)
        # e.g., if Acc is 20% at Epoch 5 while best run was 50%, stop immediately.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

def run_bayesian_optimization():
    print(f"Starting Bayesian Optimization with Optuna...")
    print(f"Device: {device}")
    
    # Create the Study
    # direction='maximize' because we want higher Accuracy
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    
    # Run the optimization
    # n_trials: Number of hyperparam combinations to try
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # --- RESULTS ---
    print("\n========================================")
    print(" BEST HYPERPARAMETERS FOUND")
    print("========================================")
    print(f"Best Trial Value (Acc): {study.best_value:.2f}%")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Optional: Visualization
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image("Experiments/Plots/optuna_history.png")
        
        # Plot parameter importance (Which param mattered more? LR or WD?)
        fig2 = plot_param_importances(study)
        fig2.write_image("Experiments/Plots/optuna_importance.png")
        print("Plots saved to Experiments/Plots/")
    except ImportError:
        print("Install 'plotly' and 'kaleido' to generate optimization plots.")

if __name__ == "__main__":
    os.makedirs("Experiments/Plots", exist_ok=True)
    run_bayesian_optimization()