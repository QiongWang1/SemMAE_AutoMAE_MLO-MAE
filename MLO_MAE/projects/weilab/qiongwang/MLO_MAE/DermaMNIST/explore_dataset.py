"""
DermaMNIST Dataset Explorer
Creates visualizations and statistics of the dataset
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import DermaMNIST, INFO

print("=" * 80)
print("DermaMNIST Dataset Explorer")
print("=" * 80)
print()

# Create output directory
os.makedirs('DermaMNIST/dataset_exploration', exist_ok=True)

# Get dataset info
info = INFO['dermamnist']
class_names = [info['label'][str(i)] for i in range(len(info['label']))]

print("Dataset Information:")
print("-" * 80)
print(f"Task: {info['task']}")
print(f"Number of classes: {len(class_names)}")
print(f"Image channels: {info['n_channels']}")
print(f"License: {info['license']}")
print()

print("Class Labels:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
print()

# Load datasets
print("Loading datasets...")
train_dataset = DermaMNIST(split='train', download=True, root='./data')
val_dataset = DermaMNIST(split='val', download=True, root='./data')
test_dataset = DermaMNIST(split='test', download=True, root='./data')

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
print()

# Analyze class distribution
print("Analyzing class distribution...")
train_labels = [int(train_dataset[i][1]) for i in range(len(train_dataset))]
val_labels = [int(val_dataset[i][1]) for i in range(len(val_dataset))]
test_labels = [int(test_dataset[i][1]) for i in range(len(test_dataset))]

train_dist = Counter(train_labels)
val_dist = Counter(val_labels)
test_dist = Counter(test_labels)

# Print class distribution
print("\nClass Distribution:")
print("-" * 80)
print(f"{'Class':<50} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
print("-" * 80)
for i in range(len(class_names)):
    train_count = train_dist[i]
    val_count = val_dist[i]
    test_count = test_dist[i]
    total = train_count + val_count + test_count
    print(f"{class_names[i]:<50} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
print("-" * 80)
print()

# === Visualization 1: Class Distribution ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

splits = ['Training', 'Validation', 'Test']
distributions = [train_dist, val_dist, test_dist]
colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

for idx, (split, dist, ax) in enumerate(zip(splits, distributions, axes)):
    counts = [dist[i] for i in range(len(class_names))]
    bars = ax.bar(range(len(class_names)), counts, color=colors)
    ax.set_xlabel('Disease Class', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title(f'{split} Set Distribution ({sum(counts)} samples)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(range(len(class_names)))
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('DermaMNIST/dataset_exploration/class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: class_distribution.png")
plt.close()

# === Visualization 2: Sample Images from Each Class ===
print("\nGenerating sample images...")
fig, axes = plt.subplots(len(class_names), 8, figsize=(16, 2*len(class_names)))

# Get 8 samples from each class
for class_idx in range(len(class_names)):
    # Find indices for this class in training set
    class_indices = [i for i, label in enumerate(train_labels) if label == class_idx]
    
    # Sample 8 images (or fewer if not enough samples)
    n_samples = min(8, len(class_indices))
    sample_indices = np.random.choice(class_indices, n_samples, replace=False)
    
    for col_idx, img_idx in enumerate(sample_indices):
        img, label = train_dataset[img_idx]
        img = np.array(img)
        
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[class_idx, col_idx].imshow(img)
        axes[class_idx, col_idx].axis('off')
        
        if col_idx == 0:
            # Add class name on the left
            axes[class_idx, col_idx].set_ylabel(
                f"Class {class_idx}\n{class_names[class_idx][:20]}...", 
                fontsize=9, rotation=0, ha='right', va='center'
            )
    
    # Hide unused subplots if less than 8 samples
    for col_idx in range(n_samples, 8):
        axes[class_idx, col_idx].axis('off')

plt.suptitle('Sample Images from Each Disease Class (Training Set)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('DermaMNIST/dataset_exploration/sample_images_per_class.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sample_images_per_class.png")
plt.close()

# === Visualization 3: Random Sample Grid ===
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
random_indices = np.random.choice(len(train_dataset), 64, replace=False)

for idx, ax_idx in enumerate(random_indices):
    row = idx // 8
    col = idx % 8
    
    img, label = train_dataset[ax_idx]
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    axes[row, col].set_title(f"C{int(label)}", fontsize=8)

plt.suptitle('Random Sample Grid (64 images from Training Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('DermaMNIST/dataset_exploration/random_samples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: random_samples.png")
plt.close()

# === Visualization 4: Image Statistics ===
print("\nCalculating image statistics...")
# Sample 1000 images for statistics
sample_size = min(1000, len(train_dataset))
sample_indices = np.random.choice(len(train_dataset), sample_size, replace=False)

mean_values = []
std_values = []

for idx in sample_indices:
    img, _ = train_dataset[idx]
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    mean_values.append(img.mean())
    std_values.append(img.std())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Mean distribution
axes[0].hist(mean_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(mean_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mean_values):.3f}')
axes[0].set_xlabel('Mean Pixel Value', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Mean Pixel Values', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Std distribution
axes[1].hist(std_values, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].axvline(np.mean(std_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(std_values):.3f}')
axes[1].set_xlabel('Standard Deviation', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Distribution of Pixel Standard Deviation', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('DermaMNIST/dataset_exploration/image_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: image_statistics.png")
plt.close()

# === Visualization 5: Image Size Analysis ===
sample_img, _ = train_dataset[0]
sample_img_array = np.array(sample_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original image
axes[0].imshow(sample_img_array)
axes[0].set_title(f'Original Size: {sample_img_array.shape}', fontsize=11, fontweight='bold')
axes[0].axis('off')

# After resize to 32x32 (as in our pipeline)
transform_32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
])
img_32 = transform_32(sample_img).permute(1, 2, 0).numpy()
axes[1].imshow(img_32)
axes[1].set_title(f'Resized to 32×32 (Model Input)', fontsize=11, fontweight='bold')
axes[1].axis('off')

# Show difference
axes[2].text(0.5, 0.5, 
             f'Original: {sample_img_array.shape[0]}×{sample_img_array.shape[1]}\n\n'
             f'Model Input: 32×32\n\n'
             f'Channels: {sample_img_array.shape[2]}\n\n'
             f'Total Pixels:\n'
             f'Original: {sample_img_array.shape[0] * sample_img_array.shape[1]}\n'
             f'Resized: 1024',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[2].axis('off')

plt.suptitle('Image Resizing for MLO-MAE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('DermaMNIST/dataset_exploration/image_sizing.png', dpi=300, bbox_inches='tight')
print("✓ Saved: image_sizing.png")
plt.close()

# === Save Dataset Summary ===
summary_text = f"""# DermaMNIST Dataset Summary

## Overview
- **Source**: HAM10000 (Dermatoscopic Images of Pigmented Skin Lesions)
- **Task**: Multi-class classification
- **Number of Classes**: {len(class_names)}
- **Image Channels**: {info['n_channels']} (RGB)
- **Original Size**: 28×28 pixels
- **Model Input Size**: 32×32 pixels (resized)
- **License**: {info['license']}

## Dataset Splits
- **Training**: {len(train_dataset)} samples
- **Validation**: {len(val_dataset)} samples
- **Test**: {len(test_dataset)} samples
- **Total**: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples

## Disease Classes
"""

for i, name in enumerate(class_names):
    train_count = train_dist[i]
    val_count = val_dist[i]
    test_count = test_dist[i]
    total = train_count + val_count + test_count
    summary_text += f"{i}. **{name}**\n"
    summary_text += f"   - Train: {train_count}, Val: {val_count}, Test: {test_count}, Total: {total}\n"

summary_text += f"""
## Class Balance
- Most common class: {class_names[max(train_dist, key=train_dist.get)]} ({max(train_dist.values())} training samples)
- Least common class: {class_names[min(train_dist, key=train_dist.get)]} ({min(train_dist.values())} training samples)
- Imbalance ratio: {max(train_dist.values()) / min(train_dist.values()):.2f}:1

## Image Statistics (from {sample_size} random training samples)
- Mean pixel value: {np.mean(mean_values):.3f} ± {np.std(mean_values):.3f}
- Mean std deviation: {np.mean(std_values):.3f} ± {np.std(std_values):.3f}

## Preprocessing for MLO-MAE
1. Resize from 28×28 to 32×32 (bicubic interpolation)
2. Random crop (32, padding=4) - training only
3. Random horizontal flip - training only
4. Normalize with ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Visualizations
All visualizations saved in `DermaMNIST/dataset_exploration/`:
- `class_distribution.png` - Distribution across splits
- `sample_images_per_class.png` - Example images from each class
- `random_samples.png` - Random grid of 64 images
- `image_statistics.png` - Pixel value statistics
- `image_sizing.png` - Resizing visualization

## Notes
- The dataset is moderately imbalanced (class sizes vary)
- Images show various skin lesion types with different colors, shapes, and textures
- Original HAM10000 images are much higher resolution (600×450), resized to 28×28 for MedMNIST
- Further resizing to 32×32 maintains CIFAR architecture compatibility

---
**Generated**: {np.datetime64('now')}
"""

with open('DermaMNIST/dataset_exploration/DATASET_SUMMARY.md', 'w') as f:
    f.write(summary_text)

print("✓ Saved: DATASET_SUMMARY.md")
print()

# Print summary
print("=" * 80)
print("EXPLORATION COMPLETE!")
print("=" * 80)
print()
print("Generated files in DermaMNIST/dataset_exploration/:")
print("  1. class_distribution.png       - Class distribution across splits")
print("  2. sample_images_per_class.png  - 8 examples from each class")
print("  3. random_samples.png           - 64 random training images")
print("  4. image_statistics.png         - Pixel value statistics")
print("  5. image_sizing.png             - Resizing demonstration")
print("  6. DATASET_SUMMARY.md           - Complete dataset summary")
print()
print("To view images:")
print("  - Open the PNG files in DermaMNIST/dataset_exploration/")
print("  - Read the markdown summary: cat DermaMNIST/dataset_exploration/DATASET_SUMMARY.md")
print()
print("=" * 80)

