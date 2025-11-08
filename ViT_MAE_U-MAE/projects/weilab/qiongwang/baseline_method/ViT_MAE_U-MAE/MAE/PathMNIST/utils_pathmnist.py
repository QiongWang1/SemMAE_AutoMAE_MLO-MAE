"""
Utilities for PathMNIST dataset loading and evaluation
"""
import os
import json
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST as PathMNISTDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_transforms(size: int = 32, basic_augment: bool = True, normalize: bool = True):
    """Get transforms for PathMNIST"""
    tfms = []
    tfms.append(transforms.Resize((size, size)))
    if basic_augment:
        tfms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ])
    tfms.append(transforms.ToTensor())
    if normalize:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        tfms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfms)


def get_pathmnist_dataloader(split: str,
                             batch_size: int,
                             size: int = 32,
                             download: bool = True,
                             basic_augment: bool = False,
                             normalize: bool = True,
                             num_workers: int = 4,
                             pin_memory: bool = True) -> DataLoader:
    """Get DataLoader for PathMNIST"""
    assert split in {"train", "val", "test"}
    transform = get_transforms(size=size, basic_augment=basic_augment, normalize=normalize)
    dataset = PathMNISTDataset(split=split, download=download, transform=transform, as_rgb=True)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=(split == "train"),
                      num_workers=num_workers,
                      pin_memory=pin_memory)


def verify_split_sizes(expected_train: int = 89996, expected_val: int = 10004, expected_test: int = 7180):
    """Verify PathMNIST split sizes"""
    train_len = len(PathMNISTDataset(split="train", download=True, as_rgb=True))
    val_len = len(PathMNISTDataset(split="val", download=True, as_rgb=True))
    test_len = len(PathMNISTDataset(split="test", download=True, as_rgb=True))
    if not (train_len == expected_train and val_len == expected_val and test_len == expected_test):
        raise ValueError(f"Expected (train,val,test)=({expected_train},{expected_val},{expected_test}) but got ({train_len},{val_len},{test_len})")


def get_split_info() -> Dict[str, int]:
    """Get split sizes"""
    return {
        "train": len(PathMNISTDataset(split="train", download=False, as_rgb=True)),
        "val": len(PathMNISTDataset(split="val", download=False, as_rgb=True)),
        "test": len(PathMNISTDataset(split="test", download=False, as_rgb=True)),
    }


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, n_classes: int) -> Dict[str, Any]:
    """Calculate comprehensive metrics"""
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    y_true = targets.cpu().numpy()

    acc = float(accuracy_score(y_true, preds))
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, preds, labels=list(range(n_classes)), zero_division=0
    )
    precision_macro = float(np.mean(precision_per_class))
    recall_macro = float(np.mean(recall_per_class))
    f1_macro = float(np.mean(f1_per_class))

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=list(range(n_classes))).tolist(),
    }


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Save JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

