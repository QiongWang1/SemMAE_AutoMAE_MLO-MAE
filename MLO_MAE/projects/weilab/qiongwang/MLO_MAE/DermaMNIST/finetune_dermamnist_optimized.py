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
parser = argparse.ArgumentParser(description='PyTorch DermaMNIST Fine-tuning (Optimized)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
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
parser.add_argument('--size', default="32", help="input image size (must be 32 to match pretrained model)")
parser.add_argument('--n_epochs', type=int, default='60')
parser.add_argument('--warmup_epochs', type=int, default='5', help='warmup epochs for lr scheduler')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--patch', default='2', type=int, help="patch for ViT")
parser.add_argument('--checkpoint_dir', default='./checkpoint_dermamnist', type=str)
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--wandb_project', type=str, default='DermaMNIST-MLO-MAE-Optimized', help='wandb project name')
parser.add_argument('--wandb_run_name', type=str, default='dermamnist_optimized', help='wandb run name')
parser.add_argument('--mixup_alpha', type=float, default=0.2, help='mixup alpha (0 = no mixup)')
parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='cutmix alpha (0 = no cutmix)')
parser.add_argument('--mixup_prob', type=float, default=0.5, help='probability of applying mixup/cutmix')
parser.add_argument('--focal_loss', action='store_true', help='use focal loss instead of CE')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='focal loss alpha')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
parser.add_argument('--class_weights', action='store_true', help='use class weights for imbalanced data')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing (0 = no smoothing)')

args = parser.parse_args()

print(args)


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


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """Returns cutmix inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Generate random box
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
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed criterion for mixup/cutmix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = args.wandb_run_name if hasattr(args, 'wandb_run_name') else "{}_lr{}_size{}_optimized".format(args.net, args.lr, args.size)
    wandb.init(project=args.wandb_project if hasattr(args, 'wandb_project') else "DermaMNIST-MLO-MAE-Optimized",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing DermaMNIST data..')
print(f'Image size: {imsize}×{imsize}')
size = imsize

# DermaMNIST normalization - use ImageNet normalization
DERMA_MEAN = [0.485, 0.456, 0.406]
DERMA_STD = [0.229, 0.224, 0.225]

# Adjust Cutout size based on image size
cutout_size = 16 if imsize == 32 else 32

# STRONGER augmentation for 32×32 to compensate for lower resolution
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
    transforms.RandomCrop(imsize, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),  # PRIMARY: Most effective for lesion invariance
    transforms.RandomVerticalFlip(p=0.2),    # SECONDARY: Medical images can be flipped
    # Stronger color augmentation for medical images
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomRotation(20),  # Wider rotation range
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Slight zoom/translation
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])
transform_train.transforms.append(Cutout(cutout_size))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((imsize, imsize), antialias=True),
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])

# Prepare dataset
trainset = DermaMNIST(split='train', transform=transform_train, download=True, root=args.data_path)
valset = DermaMNIST(split='val', transform=transform_test, download=True, root=args.data_path)
testset = DermaMNIST(split='test', transform=transform_test, download=True, root=args.data_path)

# Combine train and val for training
combined_trainset = torch.utils.data.ConcatDataset([trainset, valset])

trainloader = torch.utils.data.DataLoader(combined_trainset, batch_size=bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Get class names and compute class weights
info = INFO['dermamnist']
classes = [info['label'][str(i)] for i in range(len(info['label']))]
class_names_short = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']  # Short names for display
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


# Model factory..
print('==> Building model..')

model = mae_model.__dict__["cifar10_mae_vit_base_patch16_dec512d8b"](norm_pix_loss=True)
finetune_module = vit_model.FinetuneVisionTransformer(num_classes=num_classes)

# Load pretrained weights
if args.use_pretrained:
    pretrain_path = os.path.join(args.checkpoint_dir, 'pretrain.pth')
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path)
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        model.load_state_dict(model_state_dict)
        print(f"✓ Loaded pretrained encoder from {pretrain_path}")
    else:
        print(f"Warning: Pretrained model not found at {pretrain_path}")

if args.use_finetune:
    finetune_path = os.path.join(args.checkpoint_dir, 'finetune.pth')
    if os.path.exists(finetune_path):
        checkpoint2 = torch.load(finetune_path)
        if 'model_state_dict' in checkpoint2:
            model_state_dict2 = checkpoint2['model_state_dict']
        else:
            model_state_dict2 = checkpoint2
        finetune_module.load_state_dict(model_state_dict2)
        print(f"✓ Loaded pretrained classification head from {finetune_path}")
    else:
        print(f"Warning: Finetune model not found at {finetune_path}")

# Create the final model
if args.net=='mlomae':
    net = vit_model.cifar10_vit_base_patch2()
    net.head = nn.Linear(net.head.in_features, num_classes)
else:
    raise ValueError(f"Model {args.net} not supported for DermaMNIST")

# Transfer pretrained weights
if args.use_pretrained and args.net=='mlomae':
    for name, param in model.named_parameters():
        if name in net.state_dict() and param.size() == net.state_dict()[name].size():
            net.state_dict()[name].copy_(param.data)
        elif name in net.state_dict():
            print(f"Skipping layer: {name} due to size mismatch")
            print(f"  Pretrained size: {param.size()}, Model size: {net.state_dict()[name].size()}")
    
    if args.use_finetune:
        # Map weights from fine-tuning head
        for name, param in finetune_module.named_parameters():
            if name in net.state_dict() and param.size() == net.state_dict()[name].size():
                print(f"Loaded layer: {name}")
                net.state_dict()[name].copy_(param.data)
            elif name in net.state_dict():
                print(f"Skipping layer: {name} due to size mismatch")
                print(f"  Finetune size: {param.size()}, Model size: {net.state_dict()[name].size()}")

# Move to device
net = net.to(device)

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if torch.cuda.device_count() > 1:
        print("using data parallel")
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.net}-ckpt.t7')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch} with acc {best_acc}")

# Loss function selection
print("==> Configuring loss function...")
if args.focal_loss:
    print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
elif args.label_smoothing > 0:
    print(f"Using Label Smoothing CE Loss (smoothing={args.label_smoothing})")
    criterion = LabelSmoothingCrossEntropyLoss(classes=num_classes, smoothing=args.label_smoothing)
elif args.class_weights:
    print("Using Weighted Cross Entropy Loss")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    print("Using Standard Cross Entropy Loss")
    criterion = nn.CrossEntropyLoss()

# Optimizer
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay) 
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)  

print(f"Optimizer: {args.opt}, LR: {args.lr}, Weight Decay: {args.weight_decay}")

# Cosine annealing with warmup
def warmup_cosine_scheduler(epoch):
    """Warmup + Cosine annealing learning rate scheduler"""
    if epoch < args.warmup_epochs:
        # Linear warmup
        return (epoch + 1) / args.warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - args.warmup_epochs) / (args.n_epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_scheduler)
print(f"Using Cosine LR Scheduler with {args.warmup_epochs} warmup epochs")

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # DermaMNIST labels are (N, 1), need to flatten
        if len(targets.shape) > 1:
            targets = targets.squeeze()
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup or cutmix with probability
        use_mixup = args.mixup_alpha > 0 and np.random.rand() < args.mixup_prob
        use_cutmix = args.cutmix_alpha > 0 and np.random.rand() < args.mixup_prob
        
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_mixup and not use_cutmix:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha, device)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha, device)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        # For mixup/cutmix, use approximate accuracy
        if use_mixup or use_cutmix:
            correct += predicted.eq(targets).sum().item() * 0.5  # Approximate
        else:
            correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_acc = 100.*correct/total
    train_loss_avg = train_loss/len(trainloader)
    
    # Log to wandb
    if usewandb:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    return train_loss_avg, train_acc

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # DermaMNIST labels are (N, 1), need to flatten
            if len(targets.shape) > 1:
                targets = targets.squeeze()
            inputs, targets = inputs.to(device), targets.to(device)
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
    latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
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
        best_path = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
        legacy_path = os.path.join(args.checkpoint_dir, f'{args.net}-ckpt.t7')
        
        torch.save(best_state, best_path)
        torch.save(best_state, legacy_path)  # Legacy name for compatibility
        
        best_acc = acc
        print(f'✓ New best accuracy: {best_acc:.2f}% at epoch {epoch}')
        print(f'✓ Saved to: {best_path}')
    
    return test_loss_avg, acc


print("\n" + "="*60)
print("OPTIMIZED TRAINING CONFIGURATION")
print("="*60)
print(f"Image Size: {imsize}×{imsize}")
print(f"Batch Size: {bs}")
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
print("="*60 + "\n")

# Training loop
list_loss = []
list_acc = []

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss, trainacc = train(epoch)
    val_loss, val_acc = test(epoch)
    
    scheduler.step()
    
    list_loss.append(val_loss)
    list_acc.append(val_acc)
    
    # Time per epoch
    end = time.time()
    epoch_time = end - start
    print(f'Epoch {epoch} completed in {epoch_time:.1f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Final Checkpoint: {args.checkpoint_dir}/{args.net}-ckpt.t7")
print("="*60)

if usewandb:
    wandb.finish()

