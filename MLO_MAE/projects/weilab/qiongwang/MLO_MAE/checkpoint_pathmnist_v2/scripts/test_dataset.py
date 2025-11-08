#!/usr/bin/env python3
"""
Test script to verify CIFAR-100 dataset can be downloaded and loaded correctly.
This script will download the dataset to ./data/ directory if not already present.
"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def test_cifar100_dataset():
    """Test CIFAR-100 dataset download and loading."""
    
    print("="*60)
    print("Testing CIFAR-100 Dataset Setup")
    print("="*60)
    print()
    
    # Set data path
    data_path = './data'
    print(f"Dataset path: {os.path.abspath(data_path)}")
    print()
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    print(f"✓ Data directory created/verified: {data_path}")
    print()
    
    # Define transforms (same as in main_cifar100.py)
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    print("Loading CIFAR-100 training dataset...")
    print("(This will download ~160 MB if not already present)")
    print()
    
    try:
        # Load training data
        train_data = datasets.CIFAR100(
            data_path, 
            train=True, 
            download=True,
            transform=train_transform
        )
        
        print(f"✓ Training dataset loaded successfully")
        print(f"  - Number of training samples: {len(train_data)}")
        print()
        
        # Load test data
        test_data = datasets.CIFAR100(
            data_path, 
            train=False, 
            download=True,
            transform=test_transform
        )
        
        print(f"✓ Test dataset loaded successfully")
        print(f"  - Number of test samples: {len(test_data)}")
        print()
        
        # Test data loader
        print("Testing DataLoader...")
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Get one batch
        images, labels = next(iter(train_loader))
        
        print(f"✓ DataLoader working correctly")
        print(f"  - Batch shape: {images.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Image dtype: {images.dtype}")
        print(f"  - Labels dtype: {labels.dtype}")
        print(f"  - Image range: [{images.min():.3f}, {images.max():.3f}]")
        print()
        
        # Check class names
        print("CIFAR-100 Classes (first 10):")
        class_names = train_data.classes[:10]
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
        print(f"  ... (total 100 classes)")
        print()
        
        # Check dataset files
        print("Dataset files in ./data/:")
        cifar_path = os.path.join(data_path, 'cifar-100-python')
        if os.path.exists(cifar_path):
            files = os.listdir(cifar_path)
            for f in files:
                file_path = os.path.join(cifar_path, f)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  - {f}: {size_mb:.2f} MB")
        print()
        
        print("="*60)
        print("✓ CIFAR-100 Dataset Setup Successful!")
        print("="*60)
        print()
        print("Dataset is ready for training at:")
        print(f"  {os.path.abspath(data_path)}")
        print()
        print("You can now run the pre-training script with:")
        print("  python main_cifar100.py --data_path ./data")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cifar100_dataset()
    exit(0 if success else 1)

