"""
Test script to verify DermaMNIST data loading and preprocessing
"""
import os
import sys

# Add parent directory to path to import MLO_MAE modules if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import medmnist
from medmnist import INFO
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Get DermaMNIST info
info = INFO['dermamnist']
print("DermaMNIST Information:")
print(f"Info dict: {info}")
print(f"Task: {info['task']}")
print(f"Number of samples: {info['n_samples']}")
print(f"License: {info['license']}")

# Load the dataset
from medmnist import DermaMNIST

# Define transform to resize to 32x32 to match CIFAR dimensions
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load training and test datasets
train_dataset = DermaMNIST(split='train', transform=transform, download=True, root='./data')
val_dataset = DermaMNIST(split='val', transform=transform, download=True, root='./data')
test_dataset = DermaMNIST(split='test', transform=transform, download=True, root='./data')

print(f"\nDataset Sizes:")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Get a sample
sample_img, sample_label = train_dataset[0]
print(f"\nSample image shape: {sample_img.shape}")
print(f"Sample label: {sample_label}")
print(f"Label type: {type(sample_label)}")

# Create a data loader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
batch_imgs, batch_labels = next(iter(train_loader))
print(f"\nBatch images shape: {batch_imgs.shape}")
print(f"Batch labels shape: {batch_labels.shape}")

print("\n✓ DermaMNIST data loading successful!")
print(f"Images are resized to 32x32 to match CIFAR dimensions")
print(f"Number of classes: {len(info['label'])}")
print(f"Class labels: {info['label']}")

