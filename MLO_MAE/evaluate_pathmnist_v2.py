"""
Evaluation script for MLO-MAE on PathMNIST v2
Generates comprehensive metrics, confusion matrix, and visualizations
All outputs saved to PathMNIST_v2/Output/evaluation/
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import json
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medmnist import PathMNIST, INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import cifar_mae_model as mae_model
import vit_model

# Argument parser
parser = argparse.ArgumentParser(description='PathMNIST v2 Evaluation')
parser.add_argument('--checkpoint_dir', default='./checkpoint_pathmnist_v2', type=str,
                    help='Directory containing the trained model checkpoint')
parser.add_argument('--checkpoint_file', default='mlomae-ckpt.t7', type=str,
                    help='Checkpoint filename')
parser.add_argument('--data_path', default='/projects/weilab/qiongwang/datasets/medmnist/', type=str,
                    help='Path to dataset')
parser.add_argument('--output_dir', default='./PathMNIST_v2/Output', type=str,
                    help='Output directory for evaluation results')
parser.add_argument('--batch_size', default=100, type=int,
                    help='Batch size for evaluation')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of data loading workers')

args = parser.parse_args()

# Create output directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(args.output_dir, f'evaluation_{timestamp}')
os.makedirs(f'{OUTPUT_DIR}/metrics', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/visualizations', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/reports', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/predictions', exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print()

# Get PathMNIST class names
info = INFO['pathmnist']
CLASS_NAMES = [info['label'][str(i)] for i in range(len(info['label']))]

print("=" * 80)
print("MLO-MAE PathMNIST v2 Evaluation")
print("=" * 80)
print(f"Number of classes: {len(CLASS_NAMES)}")
print(f"Class names: {CLASS_NAMES}")
print()

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data normalization
PATH_MEAN = [0.485, 0.456, 0.406]
PATH_STD = [0.229, 0.224, 0.225]

# Load test dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(PATH_MEAN, PATH_STD),
])

test_dataset = PathMNIST(split='test', transform=test_transform, download=True, root=args.data_path)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print(f"Test samples: {len(test_dataset)}")
print()

# Load model
print("Loading model...")
checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    print("Please run fine-tuning first.")
    sys.exit(1)

# Create model
net = vit_model.cifar10_vit_base_patch2()
net.head = nn.Linear(net.head.in_features, len(CLASS_NAMES))
net = net.to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'net' in checkpoint:
    state_dict = checkpoint['net']
    best_acc = checkpoint.get('acc', 0)
    print(f"Loaded checkpoint with reported accuracy: {best_acc:.2f}%")
else:
    state_dict = checkpoint

# Handle DataParallel wrapper (remove 'module.' prefix if present)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        # Remove 'module.' prefix
        name = k[7:]
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

net.load_state_dict(new_state_dict)
print("Model loaded successfully!")
print()

# Evaluate
print("Evaluating on test set...")
net.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        # Handle PathMNIST label format
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Save predictions
np.savez(
    f'{OUTPUT_DIR}/predictions/predictions.npz',
    predictions=all_preds,
    labels=all_labels,
    probabilities=all_probs
)

# Calculate overall metrics
accuracy = accuracy_score(all_labels, all_preds) * 100
precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

print()
print("=" * 80)
print("OVERALL RESULTS")
print("=" * 80)
print(f"Test Accuracy: {accuracy:.2f}%")
print()
print("Macro-averaged metrics:")
print(f"  Precision: {precision_macro:.2f}%")
print(f"  Recall:    {recall_macro:.2f}%")
print(f"  F1 Score:  {f1_macro:.2f}%")
print()
print("Weighted-averaged metrics:")
print(f"  Precision: {precision_weighted:.2f}%")
print(f"  Recall:    {recall_weighted:.2f}%")
print(f"  F1 Score:  {f1_weighted:.2f}%")
print()

# Classification report
print("Classification Report:")
print("-" * 80)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
print(report)

# Save classification report
report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

# Calculate per-class metrics
per_class_metrics = {}
for i, class_name in enumerate(CLASS_NAMES):
    mask = all_labels == i
    if mask.sum() > 0:
        class_acc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
        class_prec = precision_score(all_labels == i, all_preds == i, zero_division=0) * 100
        class_rec = recall_score(all_labels == i, all_preds == i, zero_division=0) * 100
        class_f1 = f1_score(all_labels == i, all_preds == i, zero_division=0) * 100
        support = mask.sum()
        
        per_class_metrics[class_name] = {
            'accuracy': float(class_acc),
            'precision': float(class_prec),
            'recall': float(class_rec),
            'f1_score': float(class_f1),
            'support': int(support)
        }

with open(f'{OUTPUT_DIR}/metrics/classification_report.json', 'w') as f:
    json.dump(report_dict, f, indent=2)
print(f"Classification report saved to {OUTPUT_DIR}/metrics/classification_report.json")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix - MLO-MAE on PathMNIST v2\nAccuracy: {accuracy:.2f}%', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to {OUTPUT_DIR}/visualizations/confusion_matrix.png")
plt.close()

# Per-class accuracy
plt.figure(figsize=(14, 8))
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
colors = plt.cm.viridis(np.linspace(0, 1, len(CLASS_NAMES)))
bars = plt.bar(range(len(CLASS_NAMES)), per_class_acc, color=colors)
plt.xlabel('Disease Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy - MLO-MAE on PathMNIST v2', fontsize=14)
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
plt.ylim([0, 105])
plt.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/visualizations/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print(f"Per-class accuracy plot saved to {OUTPUT_DIR}/visualizations/per_class_accuracy.png")
plt.close()

# Per-class metrics visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
    ax = axes[idx // 2, idx % 2]
    values = [per_class_metrics[cn][metric] for cn in CLASS_NAMES]
    colors = plt.cm.viridis(np.linspace(0, 1, len(CLASS_NAMES)))
    bars = ax.bar(range(len(CLASS_NAMES)), values, color=colors)
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel(f'{metric_name} (%)', fontsize=11)
    ax.set_title(f'Per-Class {metric_name}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Per-Class Performance Metrics - PathMNIST v2', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/visualizations/per_class_metrics.png', dpi=300, bbox_inches='tight')
print(f"Per-class metrics plot saved to {OUTPUT_DIR}/visualizations/per_class_metrics.png")
plt.close()

# Save metrics to JSON
metrics = {
    'overall': {
        'accuracy': float(accuracy),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
    },
    'per_class': per_class_metrics,
    'test_samples': len(test_dataset),
    'model_checkpoint': checkpoint_path,
    'evaluation_timestamp': timestamp,
    'output_directory': OUTPUT_DIR,
}

with open(f'{OUTPUT_DIR}/metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to {OUTPUT_DIR}/metrics/metrics.json")

# Generate summary report
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Model evaluated on {len(test_dataset)} test samples")
print(f"✓ Overall accuracy: {accuracy:.2f}%")
print(f"✓ Macro F1 score: {f1_macro:.2f}%")
print(f"✓ Weighted F1 score: {f1_weighted:.2f}%")
print(f"✓ Best performing class: {CLASS_NAMES[np.argmax(per_class_acc)]} ({per_class_acc.max():.2f}%)")
print(f"✓ Worst performing class: {CLASS_NAMES[np.argmin(per_class_acc)]} ({per_class_acc.min():.2f}%)")
print()
print(f"All results saved to: {OUTPUT_DIR}/")
print("=" * 80)
print()
print("📁 Output structure:")
print(f"  {OUTPUT_DIR}/")
print(f"    ├── metrics/")
print(f"    │   ├── metrics.json")
print(f"    │   └── classification_report.json")
print(f"    ├── visualizations/")
print(f"    │   ├── confusion_matrix.png")
print(f"    │   ├── per_class_accuracy.png")
print(f"    │   └── per_class_metrics.png")
print(f"    ├── predictions/")
print(f"    │   └── predictions.npz")
print(f"    └── reports/")
print("=" * 80)

# Create a detailed markdown report
report_md = f"""# PathMNIST v2 Evaluation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model**: MLO-MAE  
**Checkpoint**: {checkpoint_path}  
**Test Samples**: {len(test_dataset)}

---

## Overall Performance

| Metric | Macro | Weighted |
|--------|-------|----------|
| **Accuracy** | - | **{accuracy:.2f}%** |
| **Precision** | {precision_macro:.2f}% | {precision_weighted:.2f}% |
| **Recall** | {recall_macro:.2f}% | {recall_weighted:.2f}% |
| **F1 Score** | {f1_macro:.2f}% | {f1_weighted:.2f}% |

---

## Per-Class Performance

| Class | Accuracy | Precision | Recall | F1 Score | Support |
|-------|----------|-----------|--------|----------|---------|
"""

for class_name in CLASS_NAMES:
    m = per_class_metrics[class_name]
    report_md += f"| {class_name} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1_score']:.2f}% | {m['support']} |\n"

report_md += f"""
---

## Best and Worst Performing Classes

- **Best**: {CLASS_NAMES[np.argmax(per_class_acc)]} ({per_class_acc.max():.2f}% accuracy)
- **Worst**: {CLASS_NAMES[np.argmin(per_class_acc)]} ({per_class_acc.min():.2f}% accuracy)

---

## Visualizations

See the `visualizations/` directory for:
- Confusion matrix
- Per-class accuracy
- Per-class metrics comparison

---

## Files

- `metrics/metrics.json` - All metrics in JSON format
- `metrics/classification_report.json` - Detailed classification report
- `predictions/predictions.npz` - Raw predictions and probabilities
- `visualizations/` - All visualization plots

"""

with open(f'{OUTPUT_DIR}/reports/EVALUATION_REPORT.md', 'w') as f:
    f.write(report_md)

print(f"\nMarkdown report saved to {OUTPUT_DIR}/reports/EVALUATION_REPORT.md")
print()
print("✓ Evaluation completed successfully!")
print("=" * 80)

