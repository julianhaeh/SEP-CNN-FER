# SEP-CNN-FER

Convolutional Neural Network (CNN) for Facial Expression Recognition (FER) task.

## Overview

This project implements multiple CNN architectures for facial expression recognition. It provides a flexible framework to experiment with different model architectures, optimizers, and hyperparameters.

## Features

- **Multiple CNN Architectures:**
  - BasicCNN: Simple 3-layer CNN
  - DeepCNN: Deeper 6-layer CNN
  - VGGLike: VGG-inspired architecture

- **Flexible Training Pipeline:**
  - Multiple optimizer options (Adam, SGD, RMSprop)
  - Learning rate scheduling (Step, Cosine, ReduceLROnPlateau)
  - Early stopping
  - Checkpoint saving

- **Comprehensive Evaluation:**
  - Accuracy, Precision, Recall, F1-Score
  - Per-class metrics
  - Confusion matrix visualization
  - Training history plots

- **Data Augmentation:**
  - Random rotation
  - Horizontal flip
  - Normalization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/julianhaeh/SEP-CNN-FER.git
cd SEP-CNN-FER
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The expected dataset structure:
```
data/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

CSV files should have two columns:
- `image_path`: Relative path to image from data directory
- `label`: Integer label (0-6 for 7 emotion classes)

Emotion labels (standard FER2013):
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## Usage

### Single Training Run

Basic training with default parameters:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py \
    --experiment_name my_experiment \
    --model_type DeepCNN \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --optimizer adam \
    --scheduler cosine
```

### Run Multiple Experiments

To run multiple experiments with different configurations:
```bash
python run_experiments.py
```

This will run 5 predefined experiments with different architectures and hyperparameters.

### Evaluation Only

To evaluate a trained model:
```bash
python train.py \
    --evaluate \
    --checkpoint_path checkpoints/fer_cnn_baseline_best.pth \
    --experiment_name evaluation
```

## Command Line Arguments

### Experiment Settings
- `--experiment_name`: Name of the experiment (default: fer_cnn_baseline)
- `--seed`: Random seed for reproducibility (default: 42)

### Data Settings
- `--data_dir`: Directory containing the dataset (default: data)
- `--image_size`: Size to resize images to (default: 48)
- `--batch_size`: Batch size for training (default: 64)
- `--num_workers`: Number of workers for data loading (default: 4)

### Model Settings
- `--model_type`: Type of CNN model (choices: BasicCNN, DeepCNN, VGGLike)
- `--dropout_rate`: Dropout rate (default: 0.5)
- `--use_batch_norm`: Use batch normalization (default: True)

### Training Settings
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--optimizer`: Optimizer to use (choices: adam, sgd, rmsprop)
- `--scheduler`: Learning rate scheduler (choices: step, cosine, plateau)
- `--early_stopping`: Use early stopping (default: True)
- `--patience`: Patience for early stopping (default: 10)

### Device Settings
- `--device`: Device to use for training (choices: cuda, cpu)

## Project Structure

```
SEP-CNN-FER/
├── config.py              # Configuration management
├── data_loader.py         # Data loading and preprocessing
├── models.py              # CNN model architectures
├── trainer.py             # Training logic
├── evaluation.py          # Evaluation metrics and visualization
├── utils.py               # Utility functions
├── train.py               # Main training script
├── run_experiments.py     # Batch experiment runner
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Output Files

After training, the following files will be generated:

### Checkpoints (saved in `checkpoints/`)
- `{experiment_name}_best.pth`: Best model checkpoint
- `{experiment_name}_epoch_{n}.pth`: Checkpoint for specific epoch

### Logs (saved in `logs/`)
- `{experiment_name}_config.json`: Experiment configuration
- `{experiment_name}_training_history.png`: Training/validation curves
- `{experiment_name}_confusion_matrix.png`: Confusion matrix heatmap

## Model Architectures

### BasicCNN
- 3 convolutional blocks (32, 64, 128 filters)
- Max pooling after each block
- 3 fully connected layers (512, 256, num_classes)
- Batch normalization and dropout

### DeepCNN
- 6 convolutional layers in 3 blocks
- Each block has 2 conv layers (64, 128, 256 filters)
- 3 fully connected layers (1024, 512, num_classes)
- Batch normalization and dropout

### VGGLike
- VGG-inspired architecture
- 6 convolutional layers in 3 blocks
- Large fully connected layers (4096, 1024, num_classes)
- Batch normalization and dropout

## Experimentation

The framework is designed to facilitate experimentation with:

1. **Different Architectures:** Try BasicCNN, DeepCNN, or VGGLike
2. **Optimizer Comparison:** Compare Adam, SGD, and RMSprop
3. **Learning Rate Scheduling:** Test different schedulers
4. **Hyperparameter Tuning:** Adjust batch size, learning rate, dropout, etc.
5. **Data Augmentation:** Enable/disable augmentation techniques

## Results

Training results include:
- Training and validation accuracy/loss curves
- Test set evaluation metrics
- Confusion matrix
- Per-class performance metrics

## License

This project is for academic and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project implements standard CNN architectures for facial expression recognition tasks.