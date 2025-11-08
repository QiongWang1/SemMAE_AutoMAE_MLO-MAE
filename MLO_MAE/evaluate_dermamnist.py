"""
Evaluation script for MLO-MAE on DermaMNIST
Generates comprehensive metrics, confusion matrix, and visualizations
All outputs saved to DermaMNIST/Output/evaluation/
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medmnist import DermaMNIST, INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import cifar_mae_model as mae_model
import vit_model

# Create output directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f'DermaMNIST/Output/evaluation_{timestamp}'
os.makedirs(f'{OUTPUT_DIR}/metrics', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/visualizations', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/reports', exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print()

# Get DermaMNIST class names
info = INFO['dermamnist']
CLASS_NAMES = [info['label'][str(i)] for i in range(len(info['label']))]

print("=" * 80)
print("MLO-MAE DermaMNIST Evaluation")
print("=" * 80)
print(f"Number of classes: {len(CLASS_NAMES)}")
print(f"Class names: {CLASS_NAMES}")
print()

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data normalization
DERMA_MEAN = [0.485, 0.456, 0.406]
DERMA_STD = [0.229, 0.224, 0.225]

# Load test dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])

test_dataset = DermaMNIST(split='test', transform=test_transform, download=True, root='./data')
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

print(f"Test samples: {len(test_dataset)}")
print()

# Load model
print("Loading model...")
checkpoint_path = './checkpoint_dermamnist/mlomae-ckpt.t7'

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
        # Handle DermaMNIST label format
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

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds) * 100
f1 = f1_score(all_labels, all_preds, average='weighted') * 100

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Weighted F1 Score: {f1:.2f}%")
print()

# Classification report
print("Classification Report:")
print("-" * 80)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
print(report)

# Save classification report
report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
with open(f'{OUTPUT_DIR}/metrics/classification_report.json', 'w') as f:
    json.dump(report_dict, f, indent=2)
print(f"Classification report saved to {OUTPUT_DIR}/metrics/classification_report.json")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[name[:20] for name in CLASS_NAMES],
            yticklabels=[name[:20] for name in CLASS_NAMES])
plt.title(f'Confusion Matrix - MLO-MAE on DermaMNIST\nAccuracy: {accuracy:.2f}%', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to {OUTPUT_DIR}/visualizations/confusion_matrix.png")

# Per-class accuracy
plt.figure(figsize=(12, 6))
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
colors = plt.cm.viridis(np.linspace(0, 1, len(CLASS_NAMES)))
bars = plt.bar(range(len(CLASS_NAMES)), per_class_acc, color=colors)
plt.xlabel('Disease Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy - MLO-MAE on DermaMNIST', fontsize=14)
plt.xticks(range(len(CLASS_NAMES)), [name[:15] + '...' if len(name) > 15 else name for name in CLASS_NAMES], rotation=45, ha='right')
plt.ylim([0, 105])
plt.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/visualizations/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print(f"Per-class accuracy plot saved to {OUTPUT_DIR}/visualizations/per_class_accuracy.png")

# Save metrics to JSON
metrics = {
    'test_accuracy': float(accuracy),
    'weighted_f1': float(f1),
    'per_class_accuracy': {CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(len(CLASS_NAMES))},
    'total_samples': len(test_dataset),
    'model_checkpoint': checkpoint_path,
    'evaluation_timestamp': timestamp,
    'output_directory': OUTPUT_DIR,
}

with open(f'{OUTPUT_DIR}/metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to {OUTPUT_DIR}/metrics/metrics.json")

# Generate summary
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Model evaluated on {len(test_dataset)} test samples")
print(f"✓ Overall accuracy: {accuracy:.2f}%")
print(f"✓ Weighted F1 score: {f1:.2f}%")
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
print(f"    │   └── per_class_accuracy.png")
print(f"    └── reports/")
print("=" * 80)

