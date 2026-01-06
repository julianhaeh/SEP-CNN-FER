"""
Test script to verify the FER CNN implementation works correctly.
"""
import torch
import os
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import config
        import data_loader
        import models
        import trainer
        import evaluation
        import utils
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration creation."""
    print("\nTesting configuration...")
    try:
        from config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.experiment_name == "fer_cnn_baseline"
        assert cfg.seed == 42
        assert cfg.data.image_size == 48
        assert cfg.model.num_classes == 7
        print("✓ Configuration works correctly")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_models():
    """Test model creation."""
    print("\nTesting models...")
    try:
        from models import BasicCNN, DeepCNN, VGGLikeCNN
        
        # Test BasicCNN
        model = BasicCNN(num_classes=7)
        x = torch.randn(1, 1, 48, 48)
        y = model(x)
        assert y.shape == (1, 7), f"Expected shape (1, 7), got {y.shape}"
        print(f"✓ BasicCNN works (params: {sum(p.numel() for p in model.parameters()):,})")
        
        # Test DeepCNN
        model = DeepCNN(num_classes=7)
        y = model(x)
        assert y.shape == (1, 7), f"Expected shape (1, 7), got {y.shape}"
        print(f"✓ DeepCNN works (params: {sum(p.numel() for p in model.parameters()):,})")
        
        # Test VGGLikeCNN
        model = VGGLikeCNN(num_classes=7)
        y = model(x)
        assert y.shape == (1, 7), f"Expected shape (1, 7), got {y.shape}"
        print(f"✓ VGGLike works (params: {sum(p.numel() for p in model.parameters()):,})")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loader."""
    print("\nTesting data loader...")
    try:
        from data_loader import FERDataset, get_transforms
        from torch.utils.data import DataLoader
        
        # Create a dummy dataset
        train_transform, val_transform = get_transforms(image_size=48, augment=True)
        dataset = FERDataset("data", "train.csv", transform=train_transform)
        
        print(f"✓ Dataset created (size: {len(dataset)})")
        
        # Test dataloader only if dataset has data
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            print(f"✓ DataLoader created")
        else:
            print(f"✓ DataLoader test skipped (no data available)")
        
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """Test trainer creation."""
    print("\nTesting trainer...")
    try:
        from trainer import Trainer
        from models import BasicCNN
        from config import ExperimentConfig
        
        config = ExperimentConfig()
        model = BasicCNN()
        device = torch.device("cpu")
        
        trainer = Trainer(model, config, device)
        print("✓ Trainer created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    try:
        from utils import set_seed, count_parameters, get_device
        from models import BasicCNN
        
        # Test set_seed
        set_seed(42)
        print("✓ set_seed works")
        
        # Test count_parameters
        model = BasicCNN()
        total, trainable = count_parameters(model)
        assert total > 0
        assert trainable > 0
        print(f"✓ count_parameters works (total: {total:,}, trainable: {trainable:,})")
        
        # Test get_device
        device = get_device("cpu")
        assert str(device) == "cpu"
        print("✓ get_device works")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("FER CNN Implementation Test Suite")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_config,
        test_models,
        test_data_loader,
        test_trainer,
        test_utils
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The implementation is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
