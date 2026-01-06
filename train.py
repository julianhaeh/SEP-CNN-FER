"""
Main training script for FER CNN experiments.
"""
import argparse
import os
import torch
from config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig
from data_loader import get_dataloaders
from models import get_model
from trainer import Trainer
from evaluation import evaluate_model, print_evaluation_results, plot_confusion_matrix, plot_training_history
from utils import set_seed, get_device, print_model_info, save_experiment_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CNN for Facial Expression Recognition")
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='fer_cnn_baseline',
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--image_size', type=int, default=48,
                        help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='BasicCNN',
                        choices=['BasicCNN', 'DeepCNN', 'VGGLike'],
                        help='Type of CNN model to use')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--use_batch_norm', action='store_true', default=True,
                        help='Use batch normalization')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluate a trained model')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        seed=args.seed,
        device=args.device
    )
    
    # Update sub-configurations
    config.data.data_dir = args.data_dir
    config.data.image_size = args.image_size
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    config.model.model_type = args.model_type
    config.model.dropout_rate = args.dropout_rate
    config.model.use_batch_norm = args.use_batch_norm
    
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.optimizer = args.optimizer
    config.training.scheduler = args.scheduler
    config.training.early_stopping = args.early_stopping
    config.training.patience = args.patience
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Save configuration
    config_save_path = os.path.join(config.log_dir, f"{config.experiment_name}_config.json")
    save_experiment_config(config, config_save_path)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(config)
    print_model_info(model)
    
    if args.evaluate:
        # Evaluation mode
        if args.checkpoint_path is None:
            args.checkpoint_path = os.path.join(
                config.training.checkpoint_dir,
                f"{config.experiment_name}_best.pth"
            )
        
        print(f"\nLoading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        print("\nEvaluating model...")
        results = evaluate_model(model, test_loader, device)
        print_evaluation_results(results)
        
        # Plot confusion matrix
        cm_save_path = os.path.join(config.log_dir, f"{config.experiment_name}_confusion_matrix.png")
        plot_confusion_matrix(results['confusion_matrix'], save_path=cm_save_path)
        
    else:
        # Training mode
        print("\nStarting training...")
        trainer = Trainer(model, config, device)
        trainer.train(train_loader, val_loader)
        
        # Plot training history
        history_save_path = os.path.join(config.log_dir, f"{config.experiment_name}_training_history.png")
        plot_training_history(
            trainer.train_losses,
            trainer.val_losses,
            trainer.train_accs,
            trainer.val_accs,
            save_path=history_save_path
        )
        
        # Evaluate on test set
        print("\nEvaluating best model on test set...")
        best_checkpoint_path = os.path.join(
            config.training.checkpoint_dir,
            f"{config.experiment_name}_best.pth"
        )
        
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate_model(model, test_loader, device)
        print_evaluation_results(results)
        
        # Plot confusion matrix
        cm_save_path = os.path.join(config.log_dir, f"{config.experiment_name}_confusion_matrix.png")
        plot_confusion_matrix(results['confusion_matrix'], save_path=cm_save_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
