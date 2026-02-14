import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.init as init

# --- CUSTOM IMPORTS ---
from ModelArchitectures.clsReducedClassifierCustomVGG13Reduced import ReducedClassifierCustomVGG13Reduced
from Data.clsOurDataset import OurDataset

# --- PARAMETERS ---
EPOCHS = 75
BATCH_SIZE = 32

# Load entire dataset
trainDataLoader = DataLoader(OurDataset(split='all'), batch_size=BATCH_SIZE, shuffle=True)

# Global weights
CLASS_WEIGHTS = torch.tensor([1.03, 2.94, 1.02, 0.60, 0.91, 1.06])

USE_PRETRAINED = None # Set to None to train from scratch, otherwise will load weights from the path

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
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# --- TRAINING LOOP ---

def train_pipeline(model, criterion, optimizer, scheduler, epochs=55):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_history = []

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

        train_loss = running_loss / len(trainDataLoader)
        loss_history.append(train_loss)

        print(f"   [Epoch {epoch+1} Train Loss: {train_loss:.4f}]")

    return loss_history


def run_experiment():
    print("Starting Training on Entire Data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Experiments/Plots", exist_ok=True)

    # 1. INIT ARCHITECTURE
    model = ReducedClassifierCustomVGG13Reduced()
    model.apply(weights_init)
    print(f"   [Model Initialized: ReducedClassifierCustomVGG13Reduced]")

    if USE_PRETRAINED is not None:
        model.load_state_dict(torch.load(USE_PRETRAINED, map_location=device))
        print(f"   [Loaded Pretrained Weights from {USE_PRETRAINED}]")

    # 2. INIT LOSS
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    print(f"   [Using Weighted Cross-Entropy Loss]")

    # 3. INIT OPTIMIZER & SCHEDULER
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0079)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. TRAIN
    loss_history = train_pipeline(model, criterion, optimizer, scheduler, epochs=EPOCHS)

    # 5. PLOT TRAINING LOSS
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='', linewidth=2)
    plt.title("Training Loss (Entire Data)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    loss_plot_path = "Experiments/Plots/ReducedClassifier_Weighted_CE_EntireData_Loss.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"   [Saved Training Loss Plot to {loss_plot_path}]")

    # 6. SAVE MODEL
    os.makedirs("Experiments/Models", exist_ok=True)
    save_filename = "Experiments/Models/ReducedClassifier_Weighted_CE_EntireData.pth"
    torch.save(model.state_dict(), save_filename)
    print(f"   [Model saved to {save_filename}]")

    print("\nTraining completed.")


if __name__ == "__main__":
    run_experiment()
