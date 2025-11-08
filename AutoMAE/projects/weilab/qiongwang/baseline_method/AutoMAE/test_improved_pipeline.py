#!/usr/bin/env python
"""
Test script to verify improved evaluation pipeline is set up correctly
"""

import os
import sys
import torch

def test_imports():
    """Test that all required modules can be imported"""
    print("="*80)
    print("Testing Imports")
    print("="*80)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ torchvision import failed: {e}")
        return False
    
    try:
        from sklearn.metrics import accuracy_score
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from datasets.derma_dataset import DermaMNISTDataset
        print("✓ Custom dataset module")
    except ImportError as e:
        print(f"✗ Custom dataset import failed: {e}")
        return False
    
    try:
        import models_mae_derma
        print("✓ Custom MAE model module")
    except ImportError as e:
        print(f"✗ Custom MAE model import failed: {e}")
        return False
    
    return True


def test_file_existence():
    """Test that all required files exist"""
    print("\n" + "="*80)
    print("Testing File Existence")
    print("="*80)
    
    required_files = [
        'eval_classification_derma_improve.py',
        'compare_results.py',
        'slurm/eval_automae_derma_improve.sh',
        'datasets/derma_dataset.py',
        'models_mae_derma.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024
            print(f"✓ {file_path} ({size:.1f} KB)")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist


def test_checkpoint_existence():
    """Test that pretrained checkpoint exists"""
    print("\n" + "="*80)
    print("Testing Checkpoint Availability")
    print("="*80)
    
    checkpoint_paths = [
        'job/checkpoints/checkpoint-best.pth',
        'job/checkpoints/checkpoint-199.pth'
    ]
    
    found = False
    for ckpt in checkpoint_paths:
        if os.path.exists(ckpt):
            size = os.path.getsize(ckpt) / (1024 * 1024)
            print(f"✓ Found: {ckpt} ({size:.1f} MB)")
            found = True
        else:
            print(f"  Not found: {ckpt}")
    
    if not found:
        print("\n⚠️  No checkpoint found. You need a pretrained AutoMAE checkpoint.")
        print("   Expected location: job/checkpoints/checkpoint-best.pth")
        return False
    
    return True


def test_directories():
    """Test that output directories can be created"""
    print("\n" + "="*80)
    print("Testing Directory Structure")
    print("="*80)
    
    directories = [
        'job/checkpoints_improve',
        'job/results_improve',
        'job/logs_improve',
        'slurm'
    ]
    
    all_ok = True
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ {dir_path}/")
        except Exception as e:
            print(f"✗ Failed to create {dir_path}: {e}")
            all_ok = False
    
    return all_ok


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*80)
    print("Testing CUDA Availability")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("⚠️  CUDA not available")
        print("   The script will work but training will be slow on CPU")
        return False


def test_data_loading():
    """Test that data can be loaded"""
    print("\n" + "="*80)
    print("Testing Data Loading")
    print("="*80)
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from datasets.derma_dataset import DermaMNISTDataset
        
        print("  Loading train split...")
        dataset = DermaMNISTDataset(
            split='train',
            transform=None,
            download=True,
            target_size=32,
            data_dir='./data'
        )
        
        print(f"✓ Loaded {len(dataset)} training samples")
        
        # Test loading one sample
        img, label = dataset[0]
        print(f"✓ Sample shape: {img.size}, label: {label}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that model can be created"""
    print("\n" + "="*80)
    print("Testing Model Creation")
    print("="*80)
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import models_mae_derma as models_mae
        from eval_classification_derma_improve import MAEClassifier
        
        print("  Creating MAE model...")
        mae_model = models_mae.mae_vit_small_patch4(norm_pix_loss=False, scorer=False)
        print(f"✓ MAE model created")
        
        print("  Creating classifier...")
        classifier = MAEClassifier(mae_model, num_classes=7, dropout=0.3)
        print(f"✓ Classifier created")
        
        # Count parameters
        total_params = sum(p.numel() for p in classifier.parameters())
        trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params/1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
        
        # Test forward pass
        print("  Testing forward pass...")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = classifier(x)
        print(f"✓ Forward pass: input {x.shape} → output {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("AutoMAE Improved Pipeline Test Suite")
    print("="*80)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Files", test_file_existence),
        ("Checkpoint", test_checkpoint_existence),
        ("Directories", test_directories),
        ("CUDA", test_cuda),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run the improved evaluation:")
        print("  Option 1 (debug): python eval_classification_derma_improve.py --debug")
        print("  Option 2 (full):  sbatch slurm/eval_automae_derma_improve.sh")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the failed tests before running the evaluation.")
        print("See test output above for details.")
        print("="*80)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

