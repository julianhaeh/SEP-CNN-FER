"""
Script to run multiple experiments with different CNN architectures and hyperparameters.
"""
import os
import subprocess
import json


def run_experiment(config):
    """
    Run a single experiment with given configuration.
    
    Args:
        config: Dictionary with experiment configuration
    """
    cmd = ["python", "train.py"]
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print("\n" + "=" * 80)
    print(f"Running experiment: {config.get('experiment_name', 'unnamed')}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    
    subprocess.run(cmd)


def main():
    """Main function to run multiple experiments."""
    
    # Define experiments to run
    experiments = [
        {
            "experiment_name": "fer_basic_cnn",
            "model_type": "BasicCNN",
            "num_epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "adam",
            "scheduler": "step"
        },
        {
            "experiment_name": "fer_deep_cnn",
            "model_type": "DeepCNN",
            "num_epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "adam",
            "scheduler": "step"
        },
        {
            "experiment_name": "fer_vgglike",
            "model_type": "VGGLike",
            "num_epochs": 50,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "optimizer": "adam",
            "scheduler": "cosine"
        },
        {
            "experiment_name": "fer_basic_sgd",
            "model_type": "BasicCNN",
            "num_epochs": 50,
            "learning_rate": 0.01,
            "batch_size": 64,
            "optimizer": "sgd",
            "scheduler": "step"
        },
        {
            "experiment_name": "fer_deep_rmsprop",
            "model_type": "DeepCNN",
            "num_epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "rmsprop",
            "scheduler": "plateau"
        }
    ]
    
    # Save experiment configurations
    os.makedirs("logs", exist_ok=True)
    with open("logs/experiments_config.json", "w") as f:
        json.dump(experiments, f, indent=4)
    
    print("Experiment configurations saved to logs/experiments_config.json")
    
    # Run each experiment
    for exp_config in experiments:
        try:
            run_experiment(exp_config)
        except Exception as e:
            print(f"Error running experiment {exp_config['experiment_name']}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
