"""
PathMNIST Dataset Loader for MLO-MAE Pipeline
Provides 3×32×32 RGB tensors for 9-class colorectal cancer histopathology classification
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from medmnist import PathMNIST, INFO
from PIL import Image


class PathMNISTDataset(Dataset):
    """
    PathMNIST Dataset wrapper compatible with MLO-MAE pipeline
    - Resizes from 28×28 to 32×32
    - Converts to 3-channel RGB if needed
    - Returns (image, label) tuples
    """
    
    def __init__(self, split='train', transform=None, download=True, data_path='./data'):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
            download: whether to download data
            data_path: path to store data
        """
        self.split = split
        self.transform = transform
        
        # Load PathMNIST data
        self.dataset = PathMNIST(
            split=split,
            transform=None,  # We'll handle transforms ourselves
            download=download,
            root=data_path,
            size=28  # Original size
        )
        
        # Get dataset info
        self.info = INFO['pathmnist']
        self.n_classes = len(self.info['label'])
        assert self.n_classes == 9, f"Expected 9 classes, got {self.n_classes}"
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image and label from medmnist dataset
        img, label = self.dataset[idx]
        
        # Convert to PIL Image if numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # Ensure RGB (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Label is returned as array, extract scalar
        if isinstance(label, np.ndarray):
            label = int(label.item()) if label.size == 1 else int(label[0])
        
        return img, label


def get_pathmnist_transforms(split='train', img_size=32):
    """
    Get transforms for PathMNIST dataset matching DermaMNIST pipeline
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: target image size (default 32)
    
    Returns:
        transforms.Compose object
    """
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def build_pathmnist_dataloaders(
    batch_size=32,
    num_workers=4,
    img_size=32,
    data_path='./data',
    pin_memory=True
):
    """
    Build train/val/test dataloaders for PathMNIST
    
    Args:
        batch_size: batch size for training
        num_workers: number of worker threads
        img_size: target image size
        data_path: path to data directory
        pin_memory: whether to pin memory for faster GPU transfer
    
    Returns:
        dict with 'train', 'val', 'test' DataLoader objects
    """
    
    # Create datasets
    train_dataset = PathMNISTDataset(
        split='train',
        transform=get_pathmnist_transforms('train', img_size),
        data_path=data_path
    )
    
    val_dataset = PathMNISTDataset(
        split='val',
        transform=get_pathmnist_transforms('val', img_size),
        data_path=data_path
    )
    
    test_dataset = PathMNISTDataset(
        split='test',
        transform=get_pathmnist_transforms('test', img_size),
        data_path=data_path
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    print(f"PathMNIST DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Classes: {train_dataset.n_classes}")
    
    return dataloaders, train_dataset.n_classes


if __name__ == '__main__':
    # Test the dataset loader
    print("Testing PathMNIST Dataset Loader...")
    print("=" * 50)
    
    # Create dataset
    train_ds = PathMNISTDataset(split='train', transform=get_pathmnist_transforms('train'))
    val_ds = PathMNISTDataset(split='val', transform=get_pathmnist_transforms('val'))
    test_ds = PathMNISTDataset(split='test', transform=get_pathmnist_transforms('test'))
    
    print(f"\nDataset Splits:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  Total: {len(train_ds) + len(val_ds) + len(test_ds)} samples")
    print(f"  Classes: {train_ds.n_classes}")
    
    # Test sample
    img, label = train_ds[0]
    print(f"\nSample Image:")
    print(f"  Shape: {img.shape}")
    print(f"  Type: {img.dtype}")
    print(f"  Range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Label: {label} (type: {type(label)})")
    
    # Test dataloaders
    print("\n" + "=" * 50)
    print("Building DataLoaders...")
    dataloaders, n_classes = build_pathmnist_dataloaders(batch_size=32, num_workers=4)
    
    # Test batch
    batch_img, batch_label = next(iter(dataloaders['train']))
    print(f"\nBatch Test:")
    print(f"  Images shape: {batch_img.shape}")
    print(f"  Labels shape: {batch_label.shape}")
    print(f"  Labels: {batch_label[:10]}")
    
    print("\n✓ PathMNIST Dataset Loader Test Passed!")


