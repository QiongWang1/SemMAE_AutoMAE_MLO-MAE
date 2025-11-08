"""
U-MAE (Unified Masked Autoencoder) Baseline for PathMNIST Classification
Independent implementation for PathMNIST (9 classes)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
from datetime import datetime

from utils_pathmnist import (
    set_seed, get_pathmnist_dataloader,
    verify_split_sizes, get_split_info,
    calculate_metrics, save_json
)


class PatchEmbedding(nn.Module):
    """Patch Embedding for Vision Transformer"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_patches=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, embed_dim=192, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, embed_dim=192, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UMAE_Encoder(nn.Module):
    """U-MAE Encoder (processes visible and masked patches)"""
    def __init__(self, embed_dim=192, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1, num_patches=64):
        super().__init__()
        self.num_patches = num_patches
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward(self, x, mask):
        B = x.shape[0]
        N = self.num_patches
        
        # Create full sequence with mask tokens
        x_full = self.mask_token.repeat(B, N, 1)  # [B, N, embed_dim]
        
        # Place visible patches in the first positions
        num_visible = x.shape[1]
        x_full[:, :num_visible, :] = x
        
        # Add position embedding
        x = x_full + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        return x, mask


class UMAE_Decoder(nn.Module):
    """U-MAE Decoder (reconstructs all patches)"""
    def __init__(self, embed_dim=192, decoder_embed_dim=128, depth=4, num_heads=8, 
                 mlp_ratio=4, dropout=0.1, num_patches=64, in_channels=3, patch_size=4):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        # Project to decoder embedding dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Position embedding for decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        # Transformer blocks
        self.decoder_blocks = nn.Sequential(*[
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels)
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Initialize
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
    def forward(self, x):
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Add position embedding
        x = x + self.decoder_pos_embed
        x = self.pos_drop(x)
        
        # Decoder transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        pred = self.decoder_pred(x)
        
        return pred


class UMAE(nn.Module):
    """Unified Masked Autoencoder"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, 
                 depth=6, num_heads=8, mlp_ratio=4, decoder_depth=4, 
                 decoder_embed_dim=128, mask_ratio=0.75, num_patches=64, num_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, num_patches)
        
        # Encoder
        self.encoder = UMAE_Encoder(embed_dim, depth, num_heads, mlp_ratio, num_patches=num_patches)
        
        # Decoder
        self.decoder = UMAE_Decoder(embed_dim, decoder_embed_dim, decoder_depth, num_heads, 
                                    mlp_ratio, num_patches=num_patches, in_channels=in_channels, patch_size=patch_size)
        
        # Classification head
        self.classification_head = nn.Linear(embed_dim, num_classes)
        
    def random_masking(self, x, num_patches):
        """Random masking"""
        N = x.shape[1]
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(x.shape[0], N, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        
        mask = torch.ones(N, dtype=torch.bool, device=x.device)
        mask[ids_keep[0]] = False
        
        return x_masked, mask
    
    def forward_pretrain(self, images):
        """Forward pass for pretraining (reconstruction)"""
        B = images.shape[0]
        
        # Patch embedding
        patches = self.patch_embed(images)  # [B, N, embed_dim]
        
        # Random masking
        patches_masked, mask = self.random_masking(patches, self.num_patches)
        
        # Encode all patches
        encoded_patches, _ = self.encoder(patches_masked, mask)
        
        # Decode all patches
        pred_patches = self.decoder(encoded_patches)
        
        return pred_patches, mask
    
    def forward_classify(self, images):
        """Forward pass for classification"""
        # Patch embedding
        patches = self.patch_embed(images)
        
        # No masking for classification - use all patches
        mask = torch.zeros(self.num_patches, dtype=torch.bool, device=patches.device)
        
        # Encode all patches
        encoded_patches, _ = self.encoder(patches, mask)
        
        # Global average pooling
        pooled = encoded_patches.mean(dim=1)
        
        # Classification
        logits = self.classification_head(pooled)
        
        return logits


def pretrain_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Pretrain for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Pretrain]')
    for batch_idx, (data, targets) in enumerate(pbar):
        data = data.to(device)
        
        optimizer.zero_grad()
        pred_patches, mask = model.forward_pretrain(data)
        
        B = pred_patches.shape[0]
        N = pred_patches.shape[1]
        
        # Select predictions for masked patches only
        pred_masked = pred_patches[:, mask, :].reshape(B, -1)
        
        # Get target patches
        patch_size = model.patch_size
        target_patches = []
        for i in range(B):
            patches = []
            for y in range(0, data.shape[2], patch_size):
                for x in range(0, data.shape[3], patch_size):
                    patch = data[i, :, y:y+patch_size, x:x+patch_size]
                    patches.append(patch.flatten())
            target_patches.append(torch.stack(patches))
        target_patches = torch.stack(target_patches).to(device)
        
        # Select masked patches as targets
        target_masked = target_patches[:, mask, :].reshape(B, -1)
        
        # Normalize targets
        target_masked = (target_masked - target_masked.mean()) / (target_masked.std() + 1e-6)
        
        loss = criterion(pred_masked, target_masked)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def finetune_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Finetune for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Finetune]')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device).squeeze().long()
        
        optimizer.zero_grad()
        outputs = model.forward_classify(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='[Val]')
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device).squeeze().long()
            outputs = model.forward_classify(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return epoch_loss, epoch_acc, all_outputs, all_targets


def test(model, test_loader, criterion, device):
    """Test model"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='[Test]')
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device).squeeze().long()
            outputs = model.forward_classify(data)
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_outputs, all_targets


def main():
    parser = argparse.ArgumentParser(description="U-MAE Baseline for PathMNIST")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--pretrain-epochs', type=int, default=50)
    parser.add_argument('--finetune-epochs', type=int, default=1000)
    parser.add_argument('--pretrain-lr', type=float, default=1.5e-4)
    parser.add_argument('--finetune-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--embed-dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--decoder-depth', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mask-ratio', type=float, default=0.75)
    parser.add_argument('--num-classes', type=int, default=9)
    # Output directories (absolute paths under job/<JOB_ID>/)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--events-dir', type=str, required=True)
    parser.add_argument('--preds-dir', type=str, required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.events_dir, exist_ok=True)
    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print('\n' + '='*80)
    print('U-MAE PathMNIST Baseline')
    print('='*80)
    print(f'Device: {device}')
    
    print('Verifying dataset splits...')
    verify_split_sizes()
    split_info = get_split_info()
    print(f'✓ Split sizes: train={split_info["train"]} val={split_info["val"]} test={split_info["test"]}')
    
    # Data loaders
    train_loader = get_pathmnist_dataloader('train', args.batch_size, size=args.img_size, download=True, basic_augment=True)
    val_loader = get_pathmnist_dataloader('val', args.batch_size, size=args.img_size, download=False, basic_augment=False)
    test_loader = get_pathmnist_dataloader('test', args.batch_size, size=args.img_size, download=False, basic_augment=False)
    
    # Model
    num_patches = (args.img_size // args.patch_size) ** 2
    model = UMAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mask_ratio=args.mask_ratio,
        num_patches=num_patches,
        num_classes=args.num_classes
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.events_dir, 'tensorboard'))
    
    # ============= PRETRAINING =============
    print('\n' + '='*80)
    print('PHASE 1: PRETRAINING')
    print('='*80)
    
    criterion_mse = nn.MSELoss()
    optimizer_pretrain = optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
    scheduler_pretrain = CosineAnnealingLR(optimizer_pretrain, T_max=args.pretrain_epochs)
    
    for epoch in range(1, args.pretrain_epochs + 1):
        pretrain_loss = pretrain_epoch(model, train_loader, criterion_mse, optimizer_pretrain, device, epoch)
        scheduler_pretrain.step()
        
        writer.add_scalar('Pretrain/Loss', pretrain_loss, epoch)
        writer.add_scalar('Pretrain/LR', optimizer_pretrain.param_groups[0]['lr'], epoch)
        
        print(f'Pretrain Epoch {epoch}/{args.pretrain_epochs} - Loss: {pretrain_loss:.4f}')
    
    # Save pretrained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_pretrain.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }, os.path.join(args.ckpt_dir, 'pretrained.pth'))
    print(f'✓ Saved pretrained checkpoint')
    
    # ============= FINETUNING =============
    print('\n' + '='*80)
    print('PHASE 2: FINETUNING')
    print('='*80)
    
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_finetune = optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    scheduler_finetune = CosineAnnealingLR(optimizer_finetune, T_max=args.finetune_epochs)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(1, args.finetune_epochs + 1):
        # Finetune
        train_loss, train_acc = finetune_epoch(model, train_loader, criterion_ce, optimizer_finetune, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_outputs, val_targets = validate(model, val_loader, criterion_ce, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler_finetune.step()
        
        writer.add_scalar('Finetune/Train/Loss', train_loss, epoch)
        writer.add_scalar('Finetune/Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Finetune/Val/Loss', val_loss, epoch)
        writer.add_scalar('Finetune/Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Finetune/LR', optimizer_finetune.param_groups[0]['lr'], epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_finetune.state_dict(),
                'val_acc': float(val_acc),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'timestamp': datetime.now().isoformat(),
            }, os.path.join(args.ckpt_dir, 'best_checkpoint.pth'))
            print(f'✓ New best model saved at epoch {epoch} (val_acc: {val_acc:.2f}%)')
        
        print(f'Epoch {epoch}/{args.finetune_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    writer.close()
    
    # Test
    print('\n' + '='*80)
    print('FINAL EVALUATION ON TEST SET')
    print('='*80)
    
    checkpoint = torch.load(os.path.join(args.ckpt_dir, 'best_checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch')}, val_acc={checkpoint.get('val_acc'):.2f}%")
    
    test_outputs, test_targets = test(model, test_loader, criterion_ce, device)
    
    # Save predictions
    preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
    torch.save(test_outputs.cpu(), os.path.join(args.preds_dir, 'test_logits.pt'))
    np.save(os.path.join(args.preds_dir, 'test_logits.npy'), test_outputs.cpu().numpy())
    np.save(os.path.join(args.preds_dir, 'test_preds.npy'), preds)
    
    # Calculate metrics
    metrics = calculate_metrics(test_outputs, test_targets, n_classes=args.num_classes)
    metrics.update({
        'best_val_acc_percent': float(best_val_acc),
        'pretrain_epochs': int(args.pretrain_epochs),
        'finetune_epochs': int(args.finetune_epochs),
        'batch_size': int(args.batch_size),
    })
    
    # Save structured results
    save_json(metrics, os.path.join(args.results_dir, 'metrics.json'))
    
    # Save compact summaries
    with open(os.path.join(args.results_dir, 'metrics_summary.csv'), 'w') as f:
        f.write('metric,value\n')
        f.write(f"accuracy,{metrics['accuracy']:.6f}\n")
        f.write(f"precision_macro,{metrics['precision_macro']:.6f}\n")
        f.write(f"recall_macro,{metrics['recall_macro']:.6f}\n")
        f.write(f"f1_macro,{metrics['f1_macro']:.6f}\n")
    
    # Per-class metrics
    with open(os.path.join(args.results_dir, 'per_class_metrics.csv'), 'w') as f:
        f.write('class,precision,recall,f1,support\n')
        for i in range(args.num_classes):
            f.write(
                f"{i},{metrics['precision_per_class'][i]:.6f},{metrics['recall_per_class'][i]:.6f},"
                f"{metrics['f1_per_class'][i]:.6f},{metrics['support'][i]}\n"
            )
    
    # Save training history
    save_json({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': int(checkpoint.get('epoch', 0)),
        'best_val_acc': float(best_val_acc),
    }, os.path.join(args.results_dir, 'training_history.json'))
    
    print('\n' + '='*80)
    print('TEST SET RESULTS (PathMNIST)')
    print('='*80)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 (Macro):        {metrics['f1_macro']:.4f}")
    print('='*80)
    print(f'Saved results to: {args.results_dir}')
    print(f'Saved predictions to: {args.preds_dir}')


if __name__ == '__main__':
    main()

