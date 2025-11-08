"""
Datasets package for AutoMAE DermaMNIST and PathMNIST
"""

from .derma_dataset import (
    DermaMNISTDataset,
    build_derma_dataset,
    get_derma_dataloaders,
    DermaMNISTDataPrefetcher,
    fast_collate_derma
)

from .path_dataset import (
    PathMNISTDataset,
    build_pathmnist_dataset,
    get_pathmnist_dataloaders,
    PathMNISTDataPrefetcher,
    fast_collate_pathmnist
)

__all__ = [
    'DermaMNISTDataset',
    'build_derma_dataset',
    'get_derma_dataloaders',
    'DermaMNISTDataPrefetcher',
    'fast_collate_derma',
    'PathMNISTDataset',
    'build_pathmnist_dataset',
    'get_pathmnist_dataloaders',
    'PathMNISTDataPrefetcher',
    'fast_collate_pathmnist'
]

