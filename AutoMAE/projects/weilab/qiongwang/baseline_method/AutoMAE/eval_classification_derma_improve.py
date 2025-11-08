"""
Improved Classification Evaluation Script for DermaMNIST
Addresses overfitting with dropout, label smoothing, class weighting, and early stopping
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add custom modules
sys.path.insert(0, os.path.dirname(__file__))
from datasets.derma_dataset import DermaMNISTDataset, get_derma_dataloaders
import models_mae_derma as models_mae
import util.misc as misc


class ImprovedClassificationHead(nn.Module):
    """
    Improved classification head with dropout and deeper architecture
    """
    def __init__(self, embed_dim=192, num_classes=7, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


class MAEClassifier(nn.Module):
    """MAE encoder + improved classification head"""
    def __init__(self, mae_model, num_classes=7, dropout=0.3):
        super().__init__()
        self.encoder = mae_model
        self.embed_dim = mae_model.cls_token.shape[-1]
        self.classifier = ImprovedClassificationHead(self.embed_dim, num_classes, dropout)
    
    def forward(self, x):
        # Extract features using MAE encoder
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        
        # Add cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        
        # Use cls token for classification
        x = x[:, 0]
        x = self.classifier(x)
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, epsilon=0.1, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
    
    def forward(self, preds, targets):
        n_classes = preds.size(1)
        log_preds = F.log_softmax(preds, dim=1)
        
        # Create smooth labels
        smooth_targets = torch.zeros_like(log_preds).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.epsilon) + self.epsilon / n_classes
        
        # Apply class weights if provided
        if self.weight is not None:
            weight_tensor = self.weight.to(preds.device)
            loss = -(smooth_targets * log_preds * weight_tensor.unsqueeze(0)).sum(dim=1)
        else:
            loss = -(smooth_targets * log_preds).sum(dim=1)
        
        return loss.mean()


def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset"""
    labels = dataset.labels
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def get_args_parser():
    parser = argparse.ArgumentParser('DermaMNIST Improved Classification', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, help='Lower learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Higher weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch4', type=str)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('--pretrained', default='', type=str,
                        help='Path to pretrained AutoMAE checkpoint')
    parser.add_argument('--finetune_mode', default='full', type=str,
                        choices=['full', 'linear', 'partial'])
    
    # Regularization
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced dataset')
    parser.set_defaults(use_class_weights=True)
    
    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--output_dir', default='./job/checkpoints_improve', type=str)
    parser.add_argument('--results_dir', default='./job/results_improve', type=str)
    parser.add_argument('--log_dir', default='./job/logs_improve', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable augmentations (for eval only)')
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(pin_mem=True)
    
    return parser


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    """Train for one epoch with improved error handling"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    # Step scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - Improved AutoMAE', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, output_dir, improvement_notes=""):
    """Save evaluation results"""
    
    # Save to markdown
    md_path = os.path.join(output_dir, 'AutoMAE_DermaMNIST_Improved_Results.md')
    with open(md_path, 'w') as f:
        f.write('# AutoMAE DermaMNIST Improved Classification Results\n\n')
        f.write(f'**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        if improvement_notes:
            f.write('## Improvements Applied\n\n')
            f.write(improvement_notes + '\n\n')
        
        f.write('## Overall Metrics\n\n')
        f.write(f'- **Accuracy:** {results["accuracy"]*100:.2f}%\n')
        f.write(f'- **Precision:** {results["precision"]*100:.2f}%\n')
        f.write(f'- **Recall:** {results["recall"]*100:.2f}%\n')
        f.write(f'- **F1 Score:** {results["f1"]*100:.2f}%\n\n')
        
        f.write('## Per-Class Metrics\n\n')
        f.write('| Class | Precision | Recall | F1 Score |\n')
        f.write('|-------|-----------|--------|----------|\n')
        for i in range(7):
            f.write(f'| {i} | {results["precision_per_class"][i]*100:.2f}% | '
                   f'{results["recall_per_class"][i]*100:.2f}% | '
                   f'{results["f1_per_class"][i]*100:.2f}% |\n')
        
        f.write('\n## Confusion Matrix\n\n')
        f.write('```\n')
        f.write(str(results['confusion_matrix']))
        f.write('\n```\n')
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('Metric,Value\n')
        f.write(f'Accuracy,{results["accuracy"]*100:.2f}\n')
        f.write(f'Precision,{results["precision"]*100:.2f}\n')
        f.write(f'Recall,{results["recall"]*100:.2f}\n')
        f.write(f'F1,{results["f1"]*100:.2f}\n')
        for i in range(7):
            f.write(f'Class_{i}_Precision,{results["precision_per_class"][i]*100:.2f}\n')
            f.write(f'Class_{i}_Recall,{results["recall_per_class"][i]*100:.2f}\n')
            f.write(f'Class_{i}_F1,{results["f1_per_class"][i]*100:.2f}\n')
    
    print(f"\nResults saved to:")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")


def main(args):
    print('='*80)
    print('DermaMNIST Improved Classification with AutoMAE')
    print('='*80)
    
    # Setup
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading DermaMNIST dataset...")
    if args.no_augmentation:
        print("⚠️  Data augmentation DISABLED")
    
    train_loader, val_loader, test_loader = get_derma_dataloaders(
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        data_dir=args.data_path,
        no_augmentation=args.no_augmentation
    )
    
    # Compute class weights
    if args.use_class_weights:
        print("\nComputing class weights for imbalanced dataset...")
        train_dataset = train_loader.dataset
        class_weights = compute_class_weights(train_dataset)
        print(f"  Class weights: {class_weights.numpy()}")
    else:
        class_weights = None
    
    # Build model
    print("\nBuilding improved model...")
    mae_model = models_mae.__dict__[args.model](norm_pix_loss=False, scorer=False)
    
    # Load pretrained weights
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'model' in checkpoint:
            mae_model.load_state_dict(checkpoint['model'], strict=False)
        else:
            mae_model.load_state_dict(checkpoint, strict=False)
    else:
        print("No pretrained weights - training from scratch")
    
    # Create improved classifier with dropout
    model = MAEClassifier(mae_model, num_classes=args.num_classes, dropout=args.dropout)
    model.to(device)
    
    # Set finetuning mode
    if args.finetune_mode == 'linear':
        print("Finetuning mode: Linear (head only)")
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.finetune_mode == 'partial':
        print("Finetuning mode: Partial (last 2 blocks + head)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.blocks[-2:].parameters():
            param.requires_grad = True
        for param in model.encoder.norm.parameters():
            param.requires_grad = True
    else:
        print("Finetuning mode: Full (all parameters)")
    
    # Setup improved training
    criterion = LabelSmoothingCrossEntropy(
        epsilon=args.label_smoothing,
        weight=class_weights
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01
    )
    
    print(f"\nModel Configuration:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"  Dropout: {args.dropout}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Class weighting: {args.use_class_weights}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Early stopping patience: {args.patience}")
    
    # Debug mode
    if args.debug:
        print("\n⚠️  DEBUG MODE ENABLED")
        args.epochs = min(args.epochs, 3)
        args.batch_size = 16
        print(f"  Reduced to {args.epochs} epochs, batch_size={args.batch_size}")
    
    # Training loop with early stopping
    print("\n" + "="*80)
    print(f"Starting Training for up to {args.epochs} epochs")
    print("="*80)
    
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Train
        use_scheduler = epoch >= args.warmup_epochs
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler if use_scheduler else None
        )
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']*100:.2f}%, Val F1: {val_results['f1']*100:.2f}%")
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc * 100)
        val_losses.append(val_results['loss'])
        val_accs.append(val_results['accuracy'] * 100)
        
        # Early stopping and model saving
        improved = False
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            best_val_f1 = val_results['f1']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = True
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_val_f1,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_classifier_improved.pth'))
            
            print(f"  ✓ Saved best model: val_acc={best_val_acc*100:.2f}%, val_f1={best_val_f1*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"  Best val_acc: {best_val_acc*100:.2f}%, Best val_f1: {best_val_f1*100:.2f}%")
            break
    
    # Plot training curves
    print("\nGenerating training curves...")
    curves_path = os.path.join(args.results_dir, 'training_curves.png')
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, curves_path)
    print(f"  Training curves saved to: {curves_path}")
    
    # Load best model and evaluate on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_results = evaluate(model, test_loader, criterion, device)
    
    # Print results
    print("\nFinal Test Results:")
    print(f"  Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"  Precision: {test_results['precision']*100:.2f}%")
    print(f"  Recall: {test_results['recall']*100:.2f}%")
    print(f"  F1 Score: {test_results['f1']*100:.2f}%")
    
    # Improvement notes
    improvement_notes = f"""
- **Dropout**: {args.dropout} added to classification head
- **Label Smoothing**: {args.label_smoothing}
- **Class Weighting**: {"Enabled" if args.use_class_weights else "Disabled"}
- **Learning Rate**: {args.lr} (reduced from 1e-3)
- **Weight Decay**: {args.weight_decay} (increased from 0.05)
- **Early Stopping**: Patience {args.patience} epochs
- **Deeper Head**: 3-layer MLP (192→512→256→7)
- **Best Epoch**: {epoch+1 - patience_counter if patience_counter >= args.patience else epoch+1}
"""
    
    # Save results
    save_results(test_results, args.results_dir, improvement_notes)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.results_dir, 'confusion_matrix_improved.png')
    plot_confusion_matrix(test_results['confusion_matrix'], cm_path)
    print(f"  Confusion matrix saved to: {cm_path}")
    
    print("\n" + "="*80)
    print("Improved Evaluation Complete!")
    print("="*80)
    
    return test_results


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

