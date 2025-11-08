"""
Evaluation script for SemMAE on DermaMNIST
Performs linear probing for classification after pretraining
"""
import argparse
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

import models_mae_derma as models_mae
from derma_dataloader import build_derma_dataset


class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen MAE encoder"""
    def __init__(self, encoder, num_classes=7, embed_dim=192):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(embed_dim, num_classes)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            # Get encoder features
            x = self.encoder.patch_embed(x)
            x = x + self.encoder.pos_embed[:, 1:, :]
            
            # Add cls token
            cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Apply Transformer blocks
            for blk in self.encoder.blocks:
                x = blk(x)
            x = self.encoder.norm(x)
            
            # Use cls token
            x = x[:, 0]
        
        # Classify
        x = self.fc(x)
        return x


def train_linear_classifier(encoder, train_loader, val_loader, device, num_epochs=50, lr=0.001, embed_dim=192):
    """Train linear classifier on frozen encoder"""
    model = LinearClassifier(encoder, num_classes=7, embed_dim=embed_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).squeeze()
                
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.fc.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    # Load best model
    model.fc.load_state_dict(best_model_state)
    return model, best_acc


def evaluate_model(model, test_loader, device):
    """Evaluate model and compute metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).squeeze()
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    class_report = classification_report(all_labels, all_preds, 
                                        target_names=[f'Class {i}' for i in range(7)],
                                        digits=4,
                                        zero_division=0)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_report': class_report,
        'conf_matrix': conf_matrix
    }


def main():
    parser = argparse.ArgumentParser('Evaluate SemMAE on DermaMNIST')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--model', default='mae_vit_small', type=str, help='Model architecture')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--embed_dim', default=192, type=int, help='Embedding dimension of encoder')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./job', type=str, help='Output directory for results')
    parser.add_argument('--linear_epochs', default=50, type=int, help='Epochs for linear probing')
    parser.add_argument('--linear_lr', default=0.001, type=float, help='Learning rate for linear probing')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Build model
    model = models_mae.__dict__[args.model](norm_pix_loss=True)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Prepare datasets
    print("Loading datasets...")
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = build_derma_dataset(split='train', download=True, transform=transform, target_size=args.input_size)
    val_dataset = build_derma_dataset(split='val', download=True, transform=transform, target_size=args.input_size)
    test_dataset = build_derma_dataset(split='test', download=True, transform=transform, target_size=args.input_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Train linear classifier
    print("\nTraining linear classifier...")
    classifier, best_val_acc = train_linear_classifier(
        model, train_loader, val_loader, device, 
        num_epochs=args.linear_epochs, lr=args.linear_lr, embed_dim=args.embed_dim
    )
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(classifier, test_loader, device)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS ON TEST SET")
    print("="*80)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(results['class_report'])
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print(results['conf_matrix'])
    
    # Save results to markdown file
    output_file = Path(args.output_dir) / 'SemMAE_DermaMNIST_Results.md'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# SemMAE on DermaMNIST - Evaluation Results\n\n")
        f.write(f"**Date:** {Path(args.checkpoint).stat().st_mtime}\n\n")
        f.write(f"**Checkpoint:** {args.checkpoint}\n\n")
        f.write(f"**Pretrain Epoch:** {checkpoint['epoch']}\n\n")
        f.write(f"**Model:** {args.model}\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {results['accuracy']:.4f} |\n")
        f.write(f"| Precision | {results['precision']:.4f} |\n")
        f.write(f"| Recall | {results['recall']:.4f} |\n")
        f.write(f"| F1-Score | {results['f1']:.4f} |\n")
        
        f.write("\n## Per-Class Metrics\n\n")
        f.write("```\n")
        f.write(results['class_report'])
        f.write("\n```\n")
        
        f.write("\n## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(str(results['conf_matrix']))
        f.write("\n```\n")
        
        f.write("\n## Dataset Split Sizes\n\n")
        f.write(f"- Training: {len(train_dataset)}\n")
        f.write(f"- Validation: {len(val_dataset)}\n")
        f.write(f"- Test: {len(test_dataset)}\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

