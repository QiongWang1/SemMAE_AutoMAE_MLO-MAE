#!/usr/bin/env python3
"""
Split and Transform Verification Script
Ensures dataset splits are correct and transforms are as expected
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from medmnist import DermaMNIST
from torchvision import transforms
import numpy as np

def check_split_sizes():
    """Verify DermaMNIST split sizes match expected values"""
    print("="*80)
    print("CHECKING DATASET SPLIT SIZES")
    print("="*80)
    
    expected = {
        'train': 7007,
        'val': 1003,
        'test': 2005
    }
    
    actual = {}
    all_ok = True
    
    for split in ['train', 'val', 'test']:
        dataset = DermaMNIST(split=split, transform=transforms.ToTensor(), download=False)
        actual[split] = len(dataset)
        
        status = "✓" if actual[split] == expected[split] else "✗"
        print(f"{status} {split:5s}: {actual[split]:5d} (expected: {expected[split]:5d})")
        
        if actual[split] != expected[split]:
            all_ok = False
    
    print("="*80)
    if all_ok:
        print("✓ ALL SPLIT SIZES CORRECT")
    else:
        print("✗ SPLIT SIZE MISMATCH DETECTED")
        return False
    
    return True

def check_transforms():
    """Check and display transforms for each split"""
    print("\n" + "="*80)
    print("CHECKING DATA TRANSFORMS")
    print("="*80)
    
    # Import the actual dataloader function
    from utils import get_dermamnist_dataloader
    
    splits_info = {
        'train': {'augment': True, 'normalize': True},
        'val': {'augment': False, 'normalize': True},
        'test': {'augment': False, 'normalize': True}
    }
    
    for split, config in splits_info.items():
        print(f"\n{split.upper()} transforms:")
        print("-" * 40)
        
        # Get a sample dataloader to inspect transforms
        loader = get_dermamnist_dataloader(
            split=split, 
            batch_size=2, 
            download=False,
            basic_augment=config['augment'],
            normalize=config['normalize']
        )
        
        # Display the transform pipeline
        dataset = loader.dataset
        transform_str = str(dataset.transform)
        print(transform_str)
        
        # Verify expected transforms
        if split == 'train' and config['augment']:
            required = ['RandomHorizontalFlip', 'RandomCrop', 'Normalize']
            for req in required:
                if req in transform_str:
                    print(f"  ✓ {req} present")
                else:
                    print(f"  ✗ {req} MISSING")
        
        if config['normalize']:
            if 'Normalize' in transform_str:
                print(f"  ✓ Normalization enabled")
            else:
                print(f"  ✗ Normalization MISSING")
    
    print("\n" + "="*80)
    print("✓ TRANSFORM CHECK COMPLETE")
    print("="*80)
    
    return True

def check_no_overlap():
    """Basic check that splits don't overlap (checks first 10 samples)"""
    print("\n" + "="*80)
    print("CHECKING FOR SPLIT OVERLAP (Sample Check)")
    print("="*80)
    
    # Load a few samples from each split
    splits = {}
    for split in ['train', 'val', 'test']:
        dataset = DermaMNIST(split=split, transform=None, download=False)
        # Get first 10 samples as arrays
        splits[split] = [dataset[i][0] for i in range(min(10, len(dataset)))]
    
    # Check if any samples look identical (simple check)
    # Note: MedMNIST uses official splits, so this is mostly a sanity check
    print("Checking first 10 samples from each split...")
    print("(MedMNIST provides official disjoint splits, this is a sanity check)")
    
    overlap_found = False
    for i, split1 in enumerate(['train', 'val', 'test']):
        for split2 in ['train', 'val', 'test'][i+1:]:
            for idx1, img1 in enumerate(splits[split1]):
                for idx2, img2 in enumerate(splits[split2]):
                    if np.array_equal(img1, img2):
                        print(f"  ✗ Potential overlap: {split1}[{idx1}] ≈ {split2}[{idx2}]")
                        overlap_found = True
    
    if not overlap_found:
        print("✓ No obvious overlap detected in sample check")
    else:
        print("✗ WARNING: Potential overlap detected!")
    
    print("="*80)
    
    return not overlap_found

def check_data_shape():
    """Verify data shape after transforms"""
    print("\n" + "="*80)
    print("CHECKING DATA SHAPES")
    print("="*80)
    
    from utils import get_dermamnist_dataloader
    
    train_loader = get_dermamnist_dataloader('train', batch_size=2, download=False)
    
    # Get one batch
    for data, targets in train_loader:
        print(f"Batch shape: {data.shape}")
        print(f"  Expected: [batch_size, 3, 32, 32]")
        print(f"  Actual:   {list(data.shape)}")
        
        if data.shape[1:] == (3, 32, 32):
            print("  ✓ Shape correct")
        else:
            print("  ✗ Shape INCORRECT")
            return False
        
        print(f"\nTarget shape: {targets.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  (Should be roughly [-2, +2] range if normalized)")
        
        break
    
    print("="*80)
    return True

def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("DATASET AND TRANSFORM VERIFICATION")
    print("DermaMNIST - ViT Baseline")
    print("="*80 + "\n")
    
    all_passed = True
    
    # Run checks
    try:
        if not check_split_sizes():
            all_passed = False
    except Exception as e:
        print(f"✗ Split size check FAILED: {e}")
        all_passed = False
    
    try:
        if not check_transforms():
            all_passed = False
    except Exception as e:
        print(f"✗ Transform check FAILED: {e}")
        all_passed = False
    
    try:
        if not check_no_overlap():
            all_passed = False
    except Exception as e:
        print(f"✗ Overlap check FAILED: {e}")
        all_passed = False
    
    try:
        if not check_data_shape():
            all_passed = False
    except Exception as e:
        print(f"✗ Data shape check FAILED: {e}")
        all_passed = False
    
    # Final verdict
    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("="*80)
        return 0
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("="*80)
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

