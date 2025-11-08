"""
PathMNIST DataLoader for SemMAE
Adapts PathMNIST dataset for MAE-style training
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class PathMNISTDataset(Dataset):
    """
    PathMNIST dataset wrapper for MAE pretraining
    Input: PathMNIST 28x28 images
    Output: Resized to 32x32 for patch-based processing
    """
    
    def __init__(self, medmnist_dataset, transform=None, target_size=32):
        """
        Args:
            medmnist_dataset: The PathMNIST dataset object
            transform: Optional transform to be applied on a sample
            target_size: Target image size (default: 32 for 8x8 patches with patch_size=4)
        """
        self.dataset = medmnist_dataset
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image and label from PathMNIST
        img, label = self.dataset[idx]
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(img, np.ndarray):
            if img.shape[0] == 3 or img.shape[0] == 1:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img.squeeze() if img.shape[2] == 1 else img)
        
        # Resize to target size
        resize_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        img = resize_transform(img)
        
        # Ensure 3 channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            img = self.transform(img)
        
        # Return image and label (label is not used in pretraining but kept for compatibility)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label[0] if len(label) > 0 else 0, dtype=torch.long)
        
        return img, label


def build_pathmnist_dataset(split='train', download=True, transform=None, target_size=32, data_path=None):
    """
    Build PathMNIST dataset
    
    Args:
        split: 'train', 'val', or 'test'
        download: Whether to download if not present
        transform: Optional transform
        target_size: Target image size
        data_path: Path to the dataset (if None, will download to default location)
    
    Returns:
        PathMNISTDataset instance
    """
    try:
        from medmnist import PathMNIST
    except ImportError:
        raise ImportError("Please install medmnist: pip install medmnist")
    
    # Load the base dataset
    medmnist_data = PathMNIST(split=split, download=download)
    
    # Wrap with our custom dataset
    dataset = PathMNISTDataset(medmnist_data, transform=transform, target_size=target_size)
    
    return dataset

