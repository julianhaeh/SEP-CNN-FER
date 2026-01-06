"""
Evaluation utilities for FER CNN models.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def evaluate_model(model, test_loader, device, emotion_labels=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: torch device
        emotion_labels: List of emotion label names
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    if emotion_labels is None:
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': cm,
        'per_class_metrics': {
            emotion_labels[i]: {
                'precision': precision_per_class[i] * 100,
                'recall': recall_per_class[i] * 100,
                'f1_score': f1_per_class[i] * 100,
                'support': int(support[i])
            }
            for i in range(len(emotion_labels))
        },
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def print_evaluation_results(results, emotion_labels=None):
    """Print evaluation results in a formatted way."""
    if emotion_labels is None:
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.2f}%")
    print(f"  Recall:    {results['recall']:.2f}%")
    print(f"  F1 Score:  {results['f1_score']:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<12}")
    print("-" * 60)
    
    for emotion in emotion_labels:
        metrics = results['per_class_metrics'][emotion]
        print(f"{emotion:<12} {metrics['precision']:>10.2f}% {metrics['recall']:>10.2f}% "
              f"{metrics['f1_score']:>10.2f}% {metrics['support']:>10}")
    
    print("=" * 60)


def plot_confusion_matrix(cm, emotion_labels=None, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        emotion_labels: List of emotion label names
        save_path: Path to save the plot
    """
    if emotion_labels is None:
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.close()
