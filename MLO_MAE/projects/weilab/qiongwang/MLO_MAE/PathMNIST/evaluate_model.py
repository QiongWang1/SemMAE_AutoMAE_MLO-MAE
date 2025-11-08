#!/usr/bin/env python3
"""
Evaluation script for PathMNIST MLO-MAE model
Computes comprehensive metrics including overall and per-class metrics
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import json
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import pandas as pd

# Add MLO_MAE root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vit_model
from medmnist import PathMNIST, INFO
import torchvision.transforms as transforms

def load_model(checkpoint_path, num_classes=9, device='cuda'):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model (same as in finetune script)
    net = vit_model.cifar10_vit_base_patch2()
    net.head = nn.Linear(net.head.in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'net' in checkpoint:
        net.load_state_dict(checkpoint['net'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with acc {checkpoint.get('acc', 0):.2f}%")
    else:
        net.load_state_dict(checkpoint)
        print("✓ Loaded model weights")
    
    net = net.to(device)
    net.eval()
    
    return net

def get_test_loader(data_path='./data', batch_size=100, num_workers=8):
    """Get test data loader"""
    print("Loading test dataset...")
    
    # PathMNIST normalization - same as training
    PATH_MEAN = [0.485, 0.456, 0.406]
    PATH_STD = [0.229, 0.224, 0.225]
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32), antialias=True),
        transforms.Normalize(PATH_MEAN, PATH_STD),
    ])
    
    testset = PathMNIST(split='test', transform=transform_test, download=True, root=data_path)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"✓ Loaded test set with {len(testset)} samples")
    return testloader, testset

def evaluate_model(model, testloader, device='cuda'):
    """Evaluate model and return predictions and true labels"""
    print("Running evaluation...")
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # Handle label format
            if len(targets.shape) > 1:
                targets = targets.squeeze()
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(targets.numpy())
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(testloader):
                print(f"  Processed {batch_idx + 1}/{len(testloader)} batches ({100*(batch_idx+1)/len(testloader):.1f}%)")
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    print(f"✓ Evaluation complete: {len(all_labels)} samples")
    return all_labels, all_preds

def compute_overall_metrics(y_true, y_pred):
    """Compute overall metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def compute_per_class_metrics(y_true, y_pred, class_names):
    """Compute per-class metrics"""
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Compute per-class accuracy (how many samples of this class were correctly classified)
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1)  # True class counts
    per_class_correct = cm.diagonal()
    per_class_accuracy = per_class_correct / support
    
    # Build per-class results
    per_class_results = []
    for i, class_name in enumerate(class_names):
        per_class_results.append({
            'class': class_name,
            'class_id': i,
            'accuracy': float(per_class_accuracy[i]),
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support[i])
        })
    
    return per_class_results

def print_evaluation_report(overall_metrics, per_class_metrics, class_names):
    """Print formatted evaluation report"""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print("-" * 80)
    print(f"  Accuracy:  {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  F1 Score:  {overall_metrics['f1']:.4f}")
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 80)
    print(f"{'Class':<20s} {'Accuracy':<12s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s} {'Support':<10s}")
    print("-" * 80)
    
    for result in per_class_metrics:
        class_name = result['class']
        print(f"{class_name:<20s} {result['accuracy']:>10.4f}  {result['precision']:>10.4f}  "
              f"{result['recall']:>10.4f}  {result['f1']:>10.4f}  {result['support']:>10d}")
    
    print("=" * 80 + "\n")
    
    # Classification report (sklearn format)
    print("CLASSIFICATION REPORT (sklearn format):")
    print("-" * 80)
    y_true = []
    y_pred = []
    for result in per_class_metrics:
        # We need to reconstruct y_true and y_pred for classification_report
        # But we can print what we have instead
        pass
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 80)
    accuracies = [r['accuracy'] for r in per_class_metrics]
    precisions = [r['precision'] for r in per_class_metrics]
    recalls = [r['recall'] for r in per_class_metrics]
    f1_scores = [r['f1'] for r in per_class_metrics]
    supports = [r['support'] for r in per_class_metrics]
    
    print(f"Per-class Accuracy - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
    print(f"Per-class Precision - Mean: {np.mean(precisions):.4f}, Std: {np.std(precisions):.4f}")
    print(f"Per-class Recall - Mean: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f}")
    print(f"Per-class F1 - Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}")
    print(f"Total samples: {sum(supports)}")
    print("=" * 80 + "\n")

def save_results(overall_metrics, per_class_metrics, output_path):
    """Save results to JSON file"""
    results = {
        'overall': overall_metrics,
        'per_class': per_class_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")

def save_results_table(overall_metrics, per_class_metrics, output_path):
    """Save results as a formatted table (CSV and Markdown)"""
    # Create DataFrame for per-class metrics
    df = pd.DataFrame(per_class_metrics)
    
    # Save CSV
    csv_path = output_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Per-class metrics saved to: {csv_path}")
    
    # Save Markdown table
    md_path = output_path.replace('.json', '.md')
    with open(md_path, 'w') as f:
        f.write("# Evaluation Metrics\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {overall_metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {overall_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {overall_metrics['recall']:.4f} |\n")
        f.write(f"| F1 Score | {overall_metrics['f1']:.4f} |\n\n")
        
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Accuracy | Precision | Recall | F1 | Support |\n")
        f.write("|-------|----------|-----------|--------|----|--------:|\n")
        for result in per_class_metrics:
            f.write(f"| {result['class']} | {result['accuracy']:.4f} | "
                   f"{result['precision']:.4f} | {result['recall']:.4f} | "
                   f"{result['f1']:.4f} | {result['support']} |\n")
    
    print(f"✓ Markdown report saved to: {md_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate PathMNIST MLO-MAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (finetune_best.pth)')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results JSON (default: same dir as checkpoint)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("  Warning: Running on CPU may be slow. Consider using GPU if available.")
    
    # Get class names
    info = INFO['pathmnist']
    class_names = [info['label'][str(i)] for i in range(len(info['label']))]
    num_classes = len(class_names)
    print(f"Dataset: PathMNIST with {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # Load model
    model = load_model(args.checkpoint, num_classes=num_classes, device=device)
    
    # Load test data
    testloader, testset = get_test_loader(args.data_path, args.batch_size, args.num_workers)
    
    # Evaluate
    y_true, y_pred = evaluate_model(model, testloader, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    overall_metrics = compute_overall_metrics(y_true, y_pred)
    per_class_metrics = compute_per_class_metrics(y_true, y_pred, class_names)
    
    # Print report
    print_evaluation_report(overall_metrics, per_class_metrics, class_names)
    
    # Save results
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(checkpoint_dir, 'evaluation_metrics.json')
    
    save_results(overall_metrics, per_class_metrics, args.output)
    save_results_table(overall_metrics, per_class_metrics, args.output)
    
    print("✓ Evaluation complete!")

if __name__ == '__main__':
    main()

