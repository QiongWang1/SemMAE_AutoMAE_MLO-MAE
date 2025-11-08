"""
Visualization utilities for PathMNIST dataset
Saves sample grids to quickly sanity-check the data
"""

import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor for visualization
    
    Args:
        tensor: normalized image tensor
        mean: normalization mean
        std: normalization std
    
    Returns:
        denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def save_sample_grid(dataloader, output_path, split_name='train', n_samples=64, denorm=True):
    """
    Save a grid of sample images from a dataloader
    
    Args:
        dataloader: PyTorch DataLoader
        output_path: path to save the image
        split_name: name of the split (for title)
        n_samples: number of samples to show (must be perfect square)
        denorm: whether to denormalize images
    """
    # Get samples
    images, labels = next(iter(dataloader))
    
    # Limit to n_samples
    images = images[:n_samples]
    labels = labels[:n_samples]
    
    # Denormalize if needed
    if denorm:
        images = torch.stack([denormalize(img) for img in images])
    
    # Clamp to [0, 1] range
    images = torch.clamp(images, 0, 1)
    
    # Create grid
    nrow = int(np.sqrt(n_samples))
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Convert to numpy for plotting
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis('off')
    ax.set_title(f'PathMNIST {split_name.capitalize()} Samples (n={n_samples})', 
                 fontsize=16, pad=20)
    
    # Add labels below the grid
    label_str = f"Labels (first {min(16, len(labels))}): {labels[:16].tolist()}"
    plt.figtext(0.5, 0.02, label_str, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {split_name} samples to: {output_path}")


def save_all_sample_grids(dataloaders, output_dir='./data/samples', n_samples=64):
    """
    Save sample grids for all splits (train, val, test)
    
    Args:
        dataloaders: dict with 'train', 'val', 'test' DataLoaders
        output_dir: directory to save images
        n_samples: number of samples per grid
    """
    print("\n" + "=" * 60)
    print("Generating Sample Visualizations for PathMNIST")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, dataloader in dataloaders.items():
        output_path = os.path.join(output_dir, f'{split_name}_samples.png')
        save_sample_grid(
            dataloader, 
            output_path, 
            split_name=split_name, 
            n_samples=n_samples,
            denorm=True
        )
    
    print("=" * 60)
    print(f"✓ All sample grids saved to: {output_dir}")
    print("=" * 60 + "\n")


def visualize_batch(images, labels, predictions=None, output_path=None, n_show=16):
    """
    Visualize a batch of images with labels and optional predictions
    
    Args:
        images: batch of images (B, C, H, W)
        labels: ground truth labels (B,)
        predictions: predicted labels (B,) or None
        output_path: path to save visualization, or None to display
        n_show: number of images to show
    """
    # Limit to n_show
    images = images[:n_show]
    labels = labels[:n_show]
    if predictions is not None:
        predictions = predictions[:n_show]
    
    # Denormalize images
    images = torch.stack([denormalize(img) for img in images])
    images = torch.clamp(images, 0, 1)
    
    # Create grid
    nrow = 4
    ncol = (n_show + nrow - 1) // nrow
    
    fig, axes = plt.subplots(ncol, nrow, figsize=(12, 3 * ncol))
    if ncol == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_show):
        row = idx // nrow
        col = idx % nrow
        ax = axes[row, col]
        
        # Show image
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis('off')
        
        # Title with label (and prediction if available)
        if predictions is not None:
            correct = "✓" if labels[idx] == predictions[idx] else "✗"
            title = f"{correct} GT:{labels[idx].item()}, Pred:{predictions[idx].item()}"
        else:
            title = f"Label: {labels[idx].item()}"
        ax.set_title(title, fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_show, nrow * ncol):
        row = idx // nrow
        col = idx % nrow
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization to: {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # Test visualization utilities
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from datasets.pathmnist_dataset import build_pathmnist_dataloaders
    
    print("Testing PathMNIST Visualization Utilities...")
    print("=" * 60)
    
    # Build dataloaders
    dataloaders, n_classes = build_pathmnist_dataloaders(
        batch_size=64,
        num_workers=4,
        img_size=32,
        data_path='./data'
    )
    
    # Save sample grids
    save_all_sample_grids(dataloaders, output_dir='./data/samples', n_samples=64)
    
    # Test batch visualization
    images, labels = next(iter(dataloaders['test']))
    visualize_batch(
        images, 
        labels, 
        output_path='./data/samples/test_batch_example.png',
        n_show=16
    )
    
    print("\n✓ Visualization utilities test passed!")


