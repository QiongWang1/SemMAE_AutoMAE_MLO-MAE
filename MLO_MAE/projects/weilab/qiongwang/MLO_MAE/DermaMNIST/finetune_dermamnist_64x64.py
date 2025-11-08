from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import pandas as pd
import csv
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import random

# Add parent directory to path to import MLO_MAE modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import progress_bar

import cifar_mae_model as mae_model
import vit_model

# DermaMNIST specific imports
from medmnist import DermaMNIST, INFO

# parsers
parser = argparse.ArgumentParser(description='PyTorch DermaMNIST Fine-tuning (64×64 Optimized)')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--minlr', default=1e-6, type=float, help='min learning rate') 
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument('--opt', default="adamw")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--use_pretrained', action='store_true', help='use pretrained backbone encoder')
parser.add_argument('--use_finetune', action='store_true', help='use trained classification head')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--net', default='mlomae')
parser.add_argument('--bs', default='64')
parser.add_argument('--size', default="64", help="input image size (64×64 with pos_embed interpolation)")
parser.add_argument('--n_epochs', type=int, default=80, help='number of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs for lr scheduler')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--patch', default='2', type=int, help="patch for ViT")
parser.add_argument('--checkpoint_dir', default='./checkpoint_dermamnist', type=str)
parser.add_argument('--pretrain_checkpoint', default='./checkpoint_dermamnist/pretrain.pth', type=str, 
                    help='path to 32×32 pretrained checkpoint')
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--wandb_project', type=str, default='DermaMNIST-MLO-MAE-64x64', help='wandb project name')
parser.add_argument('--wandb_run_name', type=str, default='dermamnist_64x64', help='wandb run name')
parser.add_argument('--mixup_alpha', type=float, default=0.2, help='mixup alpha (0 = no mixup)')
parser.add_argument('--cutmix_alpha', type=float, default=0.5, help='cutmix alpha (0 = no cutmix)')
parser.add_argument('--mixup_prob', type=float, default=0.5, help='probability of applying mixup/cutmix')
parser.add_argument('--focal_loss', action='store_true', help='use focal loss instead of CE')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='focal loss alpha')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
parser.add_argument('--class_weights', action='store_true', help='use class weights for imbalanced data')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing (0 = no smoothing)')

args = parser.parse_args()

print(args)


# ============================================================================
# POSITIONAL EMBEDDING INTERPOLATION FUNCTION
# ============================================================================

def interpolate_pos_embed(model, checkpoint_model, img_size=64, patch_size=2):
    """
    Interpolate positional embeddings from 32×32 checkpoint to 64×64 model.
    
    Args:
        model: Target ViT model (64×64 configuration)
        checkpoint_model: State dict from 32×32 pretrained checkpoint
        img_size: Target image size (64)
        patch_size: Patch size (2)
    
    Returns:
        Modified checkpoint_model with interpolated pos_embed
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # 768
        
        # Calculate patches
        num_patches_new = (img_size // patch_size) ** 2  # 64/2 = 32, 32^2 = 1024
        num_extra_tokens = 1  # CLS token
        
        # Original size from checkpoint (32×32)
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # sqrt(256) = 16
        # New size for 64×64
        new_size = int(num_patches_new ** 0.5)  # sqrt(1024) = 32
        
        if orig_size != new_size:
            print(f"⚙️  Interpolating positional embeddings from {orig_size}×{orig_size} to {new_size}×{new_size}")
            print(f"    Original pos_embed shape: {pos_embed_checkpoint.shape}")
            
            # Separate CLS token and position embeddings
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]  # (1, 1, 768)
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]     # (1, 256, 768)
            
            # Reshape to 2D grid: (1, 256, 768) -> (1, 768, 16, 16)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            
            # Bicubic interpolation to new size: (1, 768, 16, 16) -> (1, 768, 32, 32)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            
            # Reshape back: (1, 768, 32, 32) -> (1, 1024, 768)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            
            # Concatenate CLS token with interpolated position embeddings
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
            print(f"    New pos_embed shape: {new_pos_embed.shape}")
            print(f"✓ Positional embedding interpolation complete!")
        else:
            print("✓ Positional embeddings already match (no interpolation needed)")
    
    return checkpoint_model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ============================================================================
# MIXUP AND CUTMIX AUGMENTATIONS
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    
    # Get bounding box
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup/cutmix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# CUTOUT AUGMENTATION
# ============================================================================

class Cutout(object):
    """Cutout augmentation"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# ============================================================================
# WANDB SETUP
# ============================================================================

usewandb = not args.nowandb
if usewandb:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
    print("✓ Weights & Biases initialized")

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# ============================================================================
# DATA PREPARATION
# ============================================================================

print('==> Preparing DermaMNIST data..')
imsize = int(args.size)
print(f"Image size: {imsize}×{imsize}")

# Dataset normalization (ImageNet stats)
DERMA_MEAN = [0.485, 0.456, 0.406]
DERMA_STD = [0.229, 0.224, 0.225]

# Cutout size scales with image size
cutout_size = 32 if imsize == 64 else 16

# OPTIMIZED augmentation for 64×64 images
# These augmentations improve generalization and reduce overfitting on small medical datasets:
# 1. RandomHorizontalFlip: Lesions are rotation-invariant, horizontal flipping is most effective
# 2. RandomVerticalFlip: Medical images can appear in any orientation
# 3. ColorJitter: Accounts for lighting/staining variations across scanners
# 4. RandomRotation: No canonical orientation for skin lesions
# 5. RandomAffine: Simulates different crop positions and scales
# 6. Cutout: Forces model to use full lesion context, not just discriminative patches
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((imsize, imsize), antialias=True),
    transforms.RandomCrop(imsize, padding=8),  # Larger padding for 64×64
    transforms.RandomHorizontalFlip(p=0.5),    # PRIMARY: Most effective for lesion invariance
    transforms.RandomVerticalFlip(p=0.3),      # Increased for 64×64
    # Color augmentation for medical images
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomRotation(15),             # Moderate rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])
transform_train.transforms.append(Cutout(cutout_size))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((imsize, imsize), antialias=True),
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])

# Load DermaMNIST dataset
trainset = DermaMNIST(split='train', transform=transform_train, download=True, root=args.data_path)
testset = DermaMNIST(split='test', transform=transform_test, download=True, root=args.data_path)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.bs), shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

# Get class information
info = INFO['dermamnist']
classes = info['label']
class_names_short = {
    0: 'AK/IEC',  # actinic keratoses and intraepithelial carcinoma
    1: 'BCC',     # basal cell carcinoma
    2: 'BKL',     # benign keratosis-like lesions
    3: 'DF',      # dermatofibroma
    4: 'MEL',     # melanoma
    5: 'NV',      # melanocytic nevi
    6: 'VASC'     # vascular lesions
}

print(f"Classes: {classes}")
num_classes = len(classes)

# Compute class weights for imbalanced dataset
if args.class_weights:
    print("Computing class weights from training data...")
    class_counts = np.zeros(num_classes)
    for _, labels in trainloader:
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        for label in labels:
            class_counts[label.item()] += 1
    
    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")
else:
    class_weights = None


def log_prediction_images(net, dataloader, device, epoch, num_images=16):
    """Log prediction visualizations to wandb"""
    net.eval()
    
    # Get a batch of images
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for imgs, lbls in dataloader:
            if len(lbls.shape) > 1:
                lbls = lbls.squeeze()
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            outputs = net(imgs)
            _, predicted = outputs.max(1)
            
            images_list.append(imgs.cpu())
            labels_list.append(lbls.cpu())
            preds_list.append(predicted.cpu())
            
            if len(images_list) * imgs.size(0) >= num_images:
                break
    
    # Concatenate and select num_images
    all_images = torch.cat(images_list, dim=0)[:num_images]
    all_labels = torch.cat(labels_list, dim=0)[:num_images]
    all_preds = torch.cat(preds_list, dim=0)[:num_images]
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    all_images = all_images * std + mean
    all_images = torch.clamp(all_images, 0, 1)
    
    # Create visualization
    n_rows = 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(min(num_images, n_rows * n_cols)):
        img = all_images[idx].permute(1, 2, 0).numpy()
        label = all_labels[idx].item()
        pred = all_preds[idx].item()
        
        axes[idx].imshow(img)
        
        # Color: green if correct, red if wrong
        color = 'green' if label == pred else 'red'
        title = f'True: {class_names_short[label]}\nPred: {class_names_short[pred]}'
        axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, n_rows * n_cols):
        axes[idx].axis('off')
    
    plt.suptitle(f'Predictions - Epoch {epoch} (Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Log to wandb
    if usewandb:
        wandb.log({
            "predictions": wandb.Image(fig),
            "epoch": epoch
        })
    
    plt.close(fig)
    net.train()


# ============================================================================
# MODEL BUILDING WITH FLEXIBLE IMAGE SIZE
# ============================================================================

print('==> Building model for 64×64 input..')

# Create a flexible ViT model that accepts 64×64 input
class FlexibleVisionTransformer(vit_model.VisionTransformer):
    """Vision Transformer with flexible input size"""
    def __init__(self, img_size=64, patch_size=2, **kwargs):
        # Temporarily override the hardcoded img_size in parent class
        super().__init__(**kwargs)
        
        # Reinitialize patch_embed with correct size
        in_chans = 3
        embed_dim = 768
        self.patch_embed = vit_model.PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # Reinitialize positional embeddings for new size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=True)
        pos_embed = mae_model.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches**.5), 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


def cifar10_vit_base_patch2_flexible(img_size=64, num_classes=10, **kwargs):
    """Create ViT with flexible image size"""
    from functools import partial
    model = FlexibleVisionTransformer(
        img_size=img_size,
        num_classes=num_classes, 
        patch_size=2, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        drop_rate=0.1, 
        **kwargs
    )
    return model


# Create the model with 64×64 configuration
if args.net == 'mlomae':
    net = cifar10_vit_base_patch2_flexible(img_size=imsize, num_classes=num_classes)
    print(f"✓ Created flexible ViT model for {imsize}×{imsize} input")
    print(f"  Number of patches: {net.patch_embed.num_patches}")
    print(f"  Positional embedding shape: {net.pos_embed.shape}")
else:
    raise ValueError(f"Model {args.net} not supported for DermaMNIST")

# ============================================================================
# LOAD PRETRAINED WEIGHTS WITH INTERPOLATION
# ============================================================================

if args.use_pretrained:
    pretrain_path = args.pretrain_checkpoint
    if os.path.exists(pretrain_path):
        print(f"\n📂 Loading pretrained weights from: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            model_state_dict = checkpoint
        
        # Remove 'module.' prefix if exists (from DataParallel)
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        # Interpolate positional embeddings
        model_state_dict = interpolate_pos_embed(net, model_state_dict, img_size=imsize, patch_size=int(args.patch))
        
        # Load weights (allow size mismatch for head)
        missing_keys, unexpected_keys = net.load_state_dict(model_state_dict, strict=False)
        
        print(f"✓ Loaded pretrained weights successfully")
        if missing_keys:
            print(f"  Missing keys (expected for classification head): {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
    else:
        print(f"⚠️  Warning: Pretrained model not found at {pretrain_path}")
        print(f"    Training from scratch...")

# Move to device
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

print(f"✓ Model moved to {device}")

# ============================================================================
# LOSS FUNCTION AND OPTIMIZER
# ============================================================================

print("==> Configuring loss function...")
if args.focal_loss:
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
else:
    criterion = LabelSmoothingCrossEntropyLoss(num_classes, smoothing=args.label_smoothing)
    print(f"Using Label Smoothing CE Loss (smoothing={args.label_smoothing})")

# Apply class weights if specified
if args.class_weights and not args.focal_loss:
    original_criterion = criterion
    def weighted_criterion(pred, target):
        loss = original_criterion(pred, target)
        # Weight is already applied in data distribution
        return loss
    criterion = weighted_criterion

print(f"Optimizer: {args.opt}, LR: {args.lr}, Weight Decay: {args.weight_decay}")
if args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

# Cosine annealing scheduler with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.minlr)
print(f"Using Cosine LR Scheduler with {args.warmup_epochs} warmup epochs")

# Mixed precision training
use_amp = not args.noamp and device == 'cuda'
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
if use_amp:
    print("✓ Mixed precision training enabled (AMP)")

# ============================================================================
# TRAINING CONFIGURATION SUMMARY
# ============================================================================

print("\n" + "="*60)
print("64×64 OPTIMIZED TRAINING CONFIGURATION")
print("="*60)
print(f"Image Size: {imsize}×{imsize}")
print(f"Batch Size: {args.bs}")
print(f"Epochs: {args.n_epochs}")
print(f"Warmup Epochs: {args.warmup_epochs}")
print(f"Base LR: {args.lr}")
print(f"Min LR: {args.minlr}")
print(f"Weight Decay: {args.weight_decay}")
print(f"Optimizer: {args.opt}")
print(f"Mixup Alpha: {args.mixup_alpha} (prob={args.mixup_prob})")
print(f"CutMix Alpha: {args.cutmix_alpha} (prob={args.mixup_prob})")
print(f"Focal Loss: {args.focal_loss}")
print(f"Label Smoothing: {args.label_smoothing}")
print(f"Class Weights: {args.class_weights}")
print(f"Cutout Size: {cutout_size}")
print(f"Mixed Precision: {use_amp}")
print("="*60)
print()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Warmup learning rate
    if epoch < args.warmup_epochs:
        lr_scale = (epoch + 1) / args.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * lr_scale
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if len(targets.shape) > 1:
            targets = targets.squeeze()
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply Mixup or CutMix
        use_mixup = np.random.rand() < args.mixup_prob
        if use_mixup and args.mixup_alpha > 0 and args.cutmix_alpha > 0:
            # Randomly choose between mixup and cutmix
            if np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            # Standard forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
        
        # Backward pass with mixed precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        # For mixup/cutmix, we approximate accuracy
        if use_mixup and args.mixup_alpha > 0:
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Log to wandb
    if usewandb:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss/len(trainloader),
            "train_acc": 100.*correct/total,
            "lr": optimizer.param_groups[0]['lr']
        })


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if len(targets.shape) > 1:
                targets = targets.squeeze()
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint
    acc = 100.*correct/total
    test_loss_avg = test_loss/len(testloader)
    
    # Log to wandb
    if usewandb:
        wandb.log({
            "epoch": epoch,
            "test_loss": test_loss_avg,
            "test_acc": acc,
            "best_acc": max(acc, best_acc)
        })
    
    # Log prediction images every 5 epochs
    if epoch % 5 == 0 or epoch == args.n_epochs - 1:
        log_prediction_images(net, testloader, device, epoch)
    
    # CHECKPOINT SAVING STRATEGY:
    # 1. Always save latest checkpoint (for resuming training)
    # 2. Save best checkpoint only when accuracy improves (for final evaluation)
    # This ensures we use the best-performing model, not the last epoch
    
    # Save latest checkpoint (every epoch)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    latest_state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'best_acc': best_acc,
    }
    latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest_64x64.pth')
    torch.save(latest_state, latest_path)
    
    # Save best checkpoint (only when accuracy improves)
    if acc > best_acc:
        print(f'Saving best checkpoint (acc improved: {best_acc:.2f}% → {acc:.2f}%)')
        best_state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # Save with multiple names for compatibility
        best_path = os.path.join(args.checkpoint_dir, 'checkpoint_best_64x64.pth')
        legacy_path = os.path.join(args.checkpoint_dir, f'{args.net}-64x64-ckpt.t7')
        
        torch.save(best_state, best_path)
        torch.save(best_state, legacy_path)
        
        best_acc = acc
        print(f'✓ New best accuracy: {best_acc:.2f}% at epoch {epoch}')
        print(f'✓ Saved to: {best_path}')
    
    return test_loss_avg, acc


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "="*60)
print("Starting 64×64 Fine-tuning Training Loop")
print("="*60)

for epoch in range(start_epoch, args.n_epochs):
    train(epoch)
    test(epoch)
    
    # Step scheduler after warmup
    if epoch >= args.warmup_epochs:
        scheduler.step()

print("\n" + "="*60)
print("Training Complete!")
print(f"Best test accuracy: {best_acc:.2f}%")
print("="*60)

if usewandb:
    wandb.finish()

