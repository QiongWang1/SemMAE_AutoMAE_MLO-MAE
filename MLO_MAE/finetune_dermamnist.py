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

# Add parent directory to path to import MLO_MAE modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import progress_bar

import cifar_mae_model as mae_model
import vit_model

# DermaMNIST specific imports
from medmnist import DermaMNIST, INFO

# parsers
parser = argparse.ArgumentParser(description='PyTorch DermaMNIST Fine-tuning')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--minlr', default=0, type=float, help='min learning rate') 
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument('--opt', default="adamw")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--use_pretrained', action='store_true', help='use pretrained backbone encoder')
parser.add_argument('--use_finetune', action='store_true', help='use trained classification head')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--net', default='mlomae')
parser.add_argument('--bs', default='64')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--patch', default='2', type=int, help="patch for ViT")
parser.add_argument('--checkpoint_dir', default='./checkpoint_dermamnist', type=str)
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--wandb_project', type=str, default='DermaMNIST-MLO-MAE-Finetune', help='wandb project name')
parser.add_argument('--wandb_run_name', type=str, default='dermamnist_finetune', help='wandb run name')

args = parser.parse_args()

print(args)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
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



# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = args.wandb_run_name if hasattr(args, 'wandb_run_name') else "{}_lr{}_dermamnist".format(args.net, args.lr)
    wandb.init(project=args.wandb_project if hasattr(args, 'wandb_project') else "DermaMNIST-MLO-MAE",
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
size = imsize

# DermaMNIST normalization - use ImageNet normalization
DERMA_MEAN = [0.485, 0.456, 0.406]
DERMA_STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])
transform_train.transforms.append(Cutout(16))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
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

# Get class names
info = INFO['dermamnist']
classes = [info['label'][str(i)] for i in range(len(info['label']))]
class_names_short = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']  # Short names for display
print(f"Classes: {classes}")
num_classes = len(classes)

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
        print(f"Loaded pretrained encoder from {pretrain_path}")
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
        print(f"Loaded pretrained classification head from {finetune_path}")
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

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay) 
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

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
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_acc = 100.*correct/total
    if usewandb:
        wandb.log({'train_loss': train_loss/(batch_idx+1), 'train_acc': train_acc, 'epoch': epoch})
    return train_loss/(batch_idx+1)


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
    
    # Save checkpoint.
    acc = 100.*correct/total
    
    # Log prediction images every 5 epochs
    if epoch % 5 == 0 and usewandb:
        try:
            log_prediction_images(net, testloader, device, epoch, num_images=16)
        except Exception as e:
            print(f"Warning: Could not log prediction images: {e}")
    
    if usewandb:
        wandb.log({'test_loss': test_loss/(batch_idx+1), 'test_acc': acc, 'epoch': epoch})
    
    if acc > best_acc:
        print('Saving..')
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(state, os.path.join(args.checkpoint_dir, f'{args.net}-ckpt.t7'))
        best_acc = acc
    
    return test_loss/(batch_idx+1), acc

# Main training loop
print('Starting training...')
for epoch in range(start_epoch, args.n_epochs):
    train_loss = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()
    print(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Best Acc: {best_acc:.2f}%')

print(f'Training completed! Best accuracy: {best_acc:.2f}%')
if usewandb:
    wandb.log({'final_best_acc': best_acc})
    wandb.finish()

