from ModelArchitectures.clsCustomCNN import CustomCNN
from Data.clsOurDataset import OurDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

trainDataLoader = DataLoader(OurDataset(split='train'), batch_size=32, shuffle=True)
valDataLoader = DataLoader(OurDataset(split='test'), batch_size=32, shuffle=False)

EMOTION_DICT = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}

feature_config = [ 
            {'type': 'conv', 'out': 16, 'k': 3, 's': 1, 'p': 1},
            {'type': 'act'},
            {'type': 'pool'},
            {'type': 'conv', 'out': 32, 'k': 3, 's': 1, 'p': 1},
            {'type': 'pool'}
            ]

classifier_config = [
    {'type' : 'full', 'out': 128},
    {'type': 'act'},
    {'type': 'full', 'out': 6}
]

CustomCNNModel = CustomCNN(feature_config, classifier_config)

def visualize_batch(images, labels, predictions, epoch):
    """
    Plots a grid of 32 images with Predicted vs True labels and saves it.
    """
    batch_size = images.shape[0]
    fig = plt.figure(figsize=(16, 12))
    
    # Move to CPU and convert to numpy for plotting
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for i in range(min(batch_size, 32)): # Ensure we don't crash if batch < 32
        ax = fig.add_subplot(4, 8, i + 1) # 4 rows, 8 columns
        
        # Display Image (Assuming Grayscale (1, H, W))
        # Squeeze removes the channel dimension for plotting (1, 48, 48) -> (48, 48)
        img = np.squeeze(images[i])
        ax.imshow(img, cmap='gray')
        
        # Get text labels
        pred_label = EMOTION_DICT.get(predictions[i], str(predictions[i]))
        true_label = EMOTION_DICT.get(labels[i], str(labels[i]))
        
        # Color code title: Green if correct, Red if wrong
        color = 'green' if predictions[i] == labels[i] else 'red'
        
        ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle(f"Epoch {epoch} Predictions", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    filename = f"Experiments/Plots/PlotEpoch{epoch}.jpg"
    plt.savefig(filename)
    plt.close(fig) # Free memory
    print(f"Saved visualization to {filename}")


# Training and evaluation pipeline for this experiment 

def train_evaluate_pipeline(model, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        # Wrap train loader with tqdm
        # desc="Training" adds a label to the left of the bar
        train_loop = tqdm(trainDataLoader, desc="Training", leave=True)
        
        for batch in train_loop:
            
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Update the progress bar with the current running loss
            # train_loop.n is the current iteration number
            current_avg_loss = running_loss / (train_loop.n + 1)
            train_loop.set_postfix(loss=f"{current_avg_loss:.4f}")

        avg_loss = running_loss / len(trainDataLoader)

        # --- Validation Phase ---
        model.eval()
        correct = 0
        total = 0
        
        # Wrap validation loader with tqdm
        val_loop = tqdm(valDataLoader, desc="Validation", leave=True)
        
        with torch.no_grad():
            for batch in val_loop:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update the progress bar with current accuracy
                current_acc = 100 * correct / total
                val_loop.set_postfix(accuracy=f"{current_acc:.2f}%")

        final_accuracy = 100 * correct / total
        
        # Print a clean summary at the end of the epoch
        print(f"Result Epoch {epoch+1}: Train Loss: {avg_loss:.4f} | Val Accuracy: {final_accuracy:.2f}%")

# Run the pipeline
train_evaluate_pipeline(CustomCNNModel, epochs=15)

