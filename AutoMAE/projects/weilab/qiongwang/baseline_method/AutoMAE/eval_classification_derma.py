"""
Classification Evaluation Script for DermaMNIST
Fine-tunes pretrained AutoMAE encoder for 7-class classification
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
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


class ClassificationHead(nn.Module):
    """Classification head for DermaMNIST"""
    def __init__(self, embed_dim=192, num_classes=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


class MAEClassifier(nn.Module):
    """MAE encoder + classification head"""
    def __init__(self, mae_model, num_classes=7):
        super().__init__()
        self.encoder = mae_model
        self.embed_dim = mae_model.cls_token.shape[-1]
        self.classifier = ClassificationHead(self.embed_dim, num_classes)
    
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


def get_args_parser():
    parser = argparse.ArgumentParser('DermaMNIST Classification Evaluation', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch4', type=str)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--pretrained', default='', type=str,
                        help='Path to pretrained AutoMAE checkpoint')
    parser.add_argument('--finetune_mode', default='full', type=str,
                        choices=['full', 'linear', 'partial'],
                        help='Finetuning mode: full (all params), linear (head only), partial (last few blocks)')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--output_dir', default='./job/checkpoints', type=str)
    parser.add_argument('--log_dir', default='./job/logs', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable all data augmentations (use for stable evaluation)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with reduced batch size and epochs')
    parser.set_defaults(pin_mem=True)
    
    return parser


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with robust error handling"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    try:
        for batch_idx, (images, targets) in enumerate(dataloader):
            try:
                images = images.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                if batch_idx % 20 == 0:
                    print(f"  Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"\n✗ Error in batch {batch_idx}: {e}")
                print(f"  Image shape: {images.shape if 'images' in locals() else 'N/A'}")
                print(f"  Image dtype: {images.dtype if 'images' in locals() else 'N/A'}")
                raise
                
    except Exception as e:
        print(f"\n✗ Fatal error during training epoch {epoch}: {e}")
        raise
    
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


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - DermaMNIST')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(results, output_dir):
    """Save evaluation results to markdown and CSV"""
    
    # Save to markdown
    md_path = os.path.join(output_dir, 'AutoMAE_DermaMNIST_Results.md')
    with open(md_path, 'w') as f:
        f.write('# AutoMAE DermaMNIST Classification Results\n\n')
        f.write(f'**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
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
    print('DermaMNIST Classification with AutoMAE')
    print('='*80)
    
    # Setup
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading DermaMNIST dataset...")
    if args.no_augmentation:
        print("⚠️  Data augmentation DISABLED for all splits (safe evaluation mode)")
    
    try:
        train_loader, val_loader, test_loader = get_derma_dataloaders(
            batch_size=args.batch_size,
            input_size=args.input_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            data_dir=args.data_path,
            no_augmentation=args.no_augmentation
        )
        
        # Sanity check: try loading one batch
        print("\nPerforming sanity check on data loading...")
        try:
            sample_batch = next(iter(train_loader))
            print(f"✓ Successfully loaded sample batch: {sample_batch[0].shape}, dtype={sample_batch[0].dtype}")
            print(f"  Image range: [{sample_batch[0].min():.3f}, {sample_batch[0].max():.3f}]")
        except Exception as e:
            print(f"✗ Failed to load sample batch: {e}")
            raise
            
    except Exception as e:
        print(f"\n✗ Error loading DermaMNIST dataset: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Try running with --no_augmentation flag")
        print("  2. Reduce --num_workers to 0 for debugging")
        print("  3. Check that data is properly downloaded in:", args.data_path)
        raise
    
    # Build model
    print("\nBuilding model...")
    mae_model = models_mae.__dict__[args.model](norm_pix_loss=False, scorer=False)
    
    # Load pretrained weights if available
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'model' in checkpoint:
            mae_model.load_state_dict(checkpoint['model'], strict=False)
        else:
            mae_model.load_state_dict(checkpoint, strict=False)
    else:
        print("No pretrained weights - training from scratch")
    
    # Create classifier
    model = MAEClassifier(mae_model, num_classes=args.num_classes)
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
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Debug mode adjustments
    if args.debug:
        print("\n⚠️  DEBUG MODE ENABLED")
        print(f"  Original: batch_size={args.batch_size}, epochs={args.epochs}")
        args.batch_size = min(args.batch_size, 8)
        args.epochs = min(args.epochs, 2)
        print(f"  Debug: batch_size={args.batch_size}, epochs={args.epochs}")
    
    # Training loop
    print("\n" + "="*80)
    print(f"Starting Training for {args.epochs} epochs")
    print("="*80)
    
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']*100:.2f}%")
        
        scheduler.step()
        
        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(args.output_dir, 'best_classifier.pth'))
            print(f"  Saved best model with val_acc: {best_val_acc*100:.2f}%")
    
    # Load best model and evaluate on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_classifier.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, criterion, device)
    
    # Print results
    print("\nFinal Results:")
    print(f"  Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"  Precision: {test_results['precision']*100:.2f}%")
    print(f"  Recall: {test_results['recall']*100:.2f}%")
    print(f"  F1 Score: {test_results['f1']*100:.2f}%")
    
    # Save results
    save_results(test_results, args.output_dir)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_results['confusion_matrix'], cm_path)
    print(f"  Confusion matrix saved to: {cm_path}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

