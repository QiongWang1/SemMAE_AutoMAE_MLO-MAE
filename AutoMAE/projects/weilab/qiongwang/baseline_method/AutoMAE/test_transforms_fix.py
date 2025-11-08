#!/usr/bin/env python
"""
Sanity check script to test the fixed data transforms
Tests that data loading works without OverflowError
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets.derma_dataset import get_derma_dataloaders

def test_data_loading():
    """Test data loading with fixed transforms"""
    print("="*80)
    print("Testing Fixed Data Transforms")
    print("="*80)
    
    # Test 1: Load data with augmentation (training mode)
    print("\n[Test 1] Loading data WITH augmentation (training mode)...")
    try:
        train_loader, val_loader, test_loader = get_derma_dataloaders(
            batch_size=8,
            input_size=32,
            num_workers=0,  # Use 0 workers for easier debugging
            pin_memory=False,
            data_dir='./data',
            no_augmentation=False  # Enable augmentation
        )
        
        # Try to load one batch from each split
        print("\n  Testing train loader...")
        train_batch = next(iter(train_loader))
        print(f"  ✓ Train batch loaded successfully")
        print(f"    - Shape: {train_batch[0].shape}")
        print(f"    - Dtype: {train_batch[0].dtype}")
        print(f"    - Range: [{train_batch[0].min():.3f}, {train_batch[0].max():.3f}]")
        
        print("\n  Testing val loader...")
        val_batch = next(iter(val_loader))
        print(f"  ✓ Val batch loaded successfully")
        print(f"    - Shape: {val_batch[0].shape}")
        print(f"    - Dtype: {val_batch[0].dtype}")
        print(f"    - Range: [{val_batch[0].min():.3f}, {val_batch[0].max():.3f}]")
        
        print("\n  Testing test loader...")
        test_batch = next(iter(test_loader))
        print(f"  ✓ Test batch loaded successfully")
        print(f"    - Shape: {test_batch[0].shape}")
        print(f"    - Dtype: {test_batch[0].dtype}")
        print(f"    - Range: [{test_batch[0].min():.3f}, {test_batch[0].max():.3f}]")
        
        print("\n✓ Test 1 PASSED: Data loads successfully with augmentation")
        
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Load data without augmentation (evaluation mode)
    print("\n" + "="*80)
    print("[Test 2] Loading data WITHOUT augmentation (evaluation mode)...")
    try:
        train_loader, val_loader, test_loader = get_derma_dataloaders(
            batch_size=8,
            input_size=32,
            num_workers=0,
            pin_memory=False,
            data_dir='./data',
            no_augmentation=True  # Disable augmentation
        )
        
        # Try to load one batch from each split
        print("\n  Testing train loader (no aug)...")
        train_batch = next(iter(train_loader))
        print(f"  ✓ Train batch loaded successfully")
        print(f"    - Shape: {train_batch[0].shape}")
        print(f"    - Dtype: {train_batch[0].dtype}")
        print(f"    - Range: [{train_batch[0].min():.3f}, {train_batch[0].max():.3f}]")
        
        print("\n  Testing val loader (no aug)...")
        val_batch = next(iter(val_loader))
        print(f"  ✓ Val batch loaded successfully")
        print(f"    - Shape: {val_batch[0].shape}")
        print(f"    - Dtype: {val_batch[0].dtype}")
        print(f"    - Range: [{val_batch[0].min():.3f}, {val_batch[0].max():.3f}]")
        
        print("\n  Testing test loader (no aug)...")
        test_batch = next(iter(test_loader))
        print(f"  ✓ Test batch loaded successfully")
        print(f"    - Shape: {test_batch[0].shape}")
        print(f"    - Dtype: {test_batch[0].dtype}")
        print(f"    - Range: [{test_batch[0].min():.3f}, {test_batch[0].max():.3f}]")
        
        print("\n✓ Test 2 PASSED: Data loads successfully without augmentation")
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Try loading multiple batches (stress test)
    print("\n" + "="*80)
    print("[Test 3] Stress test - loading 10 batches with augmentation...")
    try:
        train_loader, _, _ = get_derma_dataloaders(
            batch_size=16,
            input_size=32,
            num_workers=0,
            pin_memory=False,
            data_dir='./data',
            no_augmentation=False
        )
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= 10:
                break
            if i % 5 == 0:
                print(f"  Batch {i+1}: shape={images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")
        
        print(f"\n✓ Test 3 PASSED: Successfully loaded 10 batches")
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nThe transform fixes are working correctly.")
    print("You can now run the full evaluation with:")
    print("  python eval_classification_derma.py --no_augmentation --debug")
    print("\nOr submit the job with:")
    print("  sbatch job/job_eval_classfication_derma/eval_automae_derma_fixed.sh")
    print("="*80)
    
    return True

if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)

