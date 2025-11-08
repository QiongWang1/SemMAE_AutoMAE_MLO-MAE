"""
DermaMNIST Dataset Loader for AutoMAE
Loads MedMNIST DermaMNIST dataset and provides PyTorch dataset interface
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Callable
import medmnist
from medmnist import DermaMNIST


class DermaMNISTDataset(Dataset):
    """
    DermaMNIST Dataset wrapper for AutoMAE pretraining
    
    Args:
        split: 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
        download: Whether to download the dataset
        target_size: Target size for images (default: 32)
    """
    
    def __init__(
        self, 
        split: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = True,
        target_size: int = 32,
        data_dir: str = './data'
    ):
        super().__init__()
        
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Load DermaMNIST dataset
        self.dataset = DermaMNIST(
            split=split,
            download=download,
            root=data_dir
        )
        
        # Get images and labels
        self.images = self.dataset.imgs  # Shape: (N, 28, 28, 3)
        self.labels = self.dataset.labels.squeeze()  # Shape: (N,)
        
        print(f"Loaded DermaMNIST {split} split:")
        print(f"  - Images: {self.images.shape}")
        print(f"  - Labels: {self.labels.shape}")
        print(f"  - Number of samples: {len(self.images)}")
        print(f"  - Number of classes: {len(np.unique(self.labels))}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and label
        img = self.images[idx]  # (28, 28, 3)
        label = int(self.labels[idx])
        
        # Convert to PIL Image for transforms
        from PIL import Image
        img = Image.fromarray(img)
        
        # Resize to target size (32x32 for AutoMAE with patch_size=4)
        if self.target_size != 28:
            resize_transform = transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
            img = resize_transform(img)
        
        # Apply additional transforms
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def build_derma_dataset(split='train', input_size=32, is_train=True, data_dir='./data', no_augmentation=False):
    """
    Build DermaMNIST dataset with appropriate transforms for AutoMAE
    
    Args:
        split: 'train', 'val', or 'test'
        input_size: Input image size (default: 32)
        is_train: Whether to apply training augmentations
        data_dir: Directory to store dataset
        no_augmentation: If True, disable all augmentations (overrides is_train)
    
    Returns:
        DermaMNIST dataset instance
    """
    
    if is_train and not no_augmentation:
        # Training augmentations - SAFE transforms only
        # NOTE: ColorJitter removed due to OverflowError in torchvision's PIL backend
        # For medical images, geometric augmentations are more appropriate anyway
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # Apply color augmentation in tensor space (safe)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # No augmentation for validation/test - only basic preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    dataset = DermaMNISTDataset(
        split=split,
        transform=transform,
        download=True,
        target_size=input_size,
        data_dir=data_dir
    )
    
    return dataset


def get_derma_dataloaders(
    batch_size=64,
    input_size=32,
    num_workers=8,
    pin_memory=True,
    data_dir='./data',
    no_augmentation=False
):
    """
    Get train, validation, and test dataloaders for DermaMNIST
    
    Args:
        batch_size: Batch size for dataloaders
        input_size: Input image size
        num_workers: Number of workers for dataloaders
        pin_memory: Whether to pin memory
        data_dir: Directory to store dataset
        no_augmentation: If True, disable all augmentations for all splits
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Build datasets
    train_dataset = build_derma_dataset('train', input_size, is_train=True, data_dir=data_dir, no_augmentation=no_augmentation)
    val_dataset = build_derma_dataset('val', input_size, is_train=False, data_dir=data_dir, no_augmentation=no_augmentation)
    test_dataset = build_derma_dataset('test', input_size, is_train=False, data_dir=data_dir, no_augmentation=no_augmentation)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


class DermaMNISTDataPrefetcher:
    """
    Data prefetcher for DermaMNIST that normalizes images on GPU
    Similar to the HDF5 ImageNet prefetcher but adapted for DermaMNIST
    """
    def __init__(self, loader):
        self._len = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # ImageNet normalization (used in original MAE)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def __len__(self):
        return self._len

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            
            # Convert PIL images to tensors and normalize
            if self.next_input.dtype == torch.uint8:
                self.next_input = self.next_input.float()
            else:
                # Images are already in [0, 1] range from PIL, scale to [0, 255]
                if self.next_input.max() <= 1.0:
                    self.next_input = self.next_input * 255.0
            
            # Normalize
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        if self.next_input is None:
            raise StopIteration
        
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        
        self.preload()
        return input, target

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def fast_collate_derma(batch, memory_format=torch.contiguous_format):
    """
    Fast collate function for DermaMNIST
    Converts PIL images to tensors efficiently
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    
    # Check if images are PIL or already tensors
    if hasattr(imgs[0], 'size'):  # PIL Image
        w, h = imgs[0].size
        tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            if nump_array.ndim < 3:
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)
            tensor[i] += torch.from_numpy(nump_array)
    else:  # Already tensor
        tensor = torch.stack(imgs, dim=0)
    
    return tensor, targets

