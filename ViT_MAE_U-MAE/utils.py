"""
Shared utilities for ViT, MAE, and U-MAE baseline models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from medmnist import INFO
import medmnist
from torch.utils.data import DataLoader
from torchvision import transforms
import random

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Get DermaMNIST dataset info
info = INFO['dermamnist']
n_classes = len(info['label'])
data_flag = 'dermamnist'
DataClass = getattr(medmnist, 'DermaMNIST')

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def get_dermamnist_dataloader(split='train', batch_size=128, download=True, size=32, 
                              basic_augment=True, normalize=True):
    """
    Get DermaMNIST dataloader with paper-aligned augmentation
    
    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch size
        download: whether to download dataset
        size: image size (will be resized to this)
        basic_augment: if True and split=='train', apply basic augmentation (flip, crop)
        normalize: if True, apply ImageNet normalization
    
    Returns:
        DataLoader
    """
    # ImageNet normalization (standard for small datasets)
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train' and basic_augment:
        # Basic augmentation for training: flip + crop (NO RandAugment/Mixup/etc.)
        transform_list = [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size, padding=4),
            transforms.ToTensor(),
        ]
    else:
        # Minimal augmentation for val/test
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    
    if normalize:
        transform_list.append(normalize_transform)
    
    dataset = DataClass(
        split=split, 
        transform=transforms.Compose(transform_list), 
        download=download
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

def calculate_metrics(outputs, targets, n_classes=7):
    """
    Calculate accuracy, precision, recall, F1
    
    Returns:
        dict with metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Overall metrics
    accuracy = accuracy_score(targets_np, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets_np, predictions, average='micro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        targets_np, predictions, average=None, zero_division=0
    )
    
    confusion = confusion_matrix(targets_np, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support': support,
        'confusion_matrix': confusion,
    }
    
    return metrics

def save_results_csv(metrics, filepath, model_name='model'):
    """
    Save results to CSV
    """
    import csv
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Overall metrics
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f'{metrics["accuracy"]:.4f}'])
        writer.writerow(['Precision (Macro)', f'{metrics["precision_macro"]:.4f}'])
        writer.writerow(['Recall (Macro)', f'{metrics["recall_macro"]:.4f}'])
        writer.writerow(['F1 (Macro)', f'{metrics["f1_macro"]:.4f}'])
        writer.writerow(['Precision (Micro)', f'{metrics["precision_micro"]:.4f}'])
        writer.writerow(['Recall (Micro)', f'{metrics["recall_micro"]:.4f}'])
        writer.writerow(['F1 (Micro)', f'{metrics["f1_micro"]:.4f}'])
        writer.writerow([])
        writer.writerow(['Class', 'Precision', 'Recall', 'F1', 'Support'])
        for i in range(len(metrics['precision_per_class'])):
            writer.writerow([
                f'Class {i}',
                f'{metrics["precision_per_class"][i]:.4f}',
                f'{metrics["recall_per_class"][i]:.4f}',
                f'{metrics["f1_per_class"][i]:.4f}',
                int(metrics["support"][i])
            ])

def plot_confusion_matrix(metrics, filepath, class_names=None):
    """
    Plot and save confusion matrix
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    cm = metrics['confusion_matrix']
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def get_split_info():
    """
    Get DermaMNIST split information
    Returns dict with split sizes
    """
    splits_info = {}
    for split in ['train', 'val', 'test']:
        dataset = DataClass(split=split, transform=transforms.ToTensor(), download=False)
        splits_info[split] = len(dataset)
    return splits_info

def verify_split_sizes(expected_train=7007, expected_val=1003, expected_test=2005):
    """
    Verify that dataset splits match expected sizes
    Raises ValueError if mismatch detected
    """
    actual = get_split_info()
    expected = {'train': expected_train, 'val': expected_val, 'test': expected_test}
    
    mismatches = []
    for split in ['train', 'val', 'test']:
        if actual[split] != expected[split]:
            mismatches.append(f"{split}: expected {expected[split]}, got {actual[split]}")
    
    if mismatches:
        raise ValueError(f"Split size mismatch detected:\n" + "\n".join(mismatches))
    
    return True

