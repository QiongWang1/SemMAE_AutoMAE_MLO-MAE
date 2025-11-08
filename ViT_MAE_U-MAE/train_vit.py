"""
ViT Baseline for DermaMNIST Classification
No pretraining - direct classification
Hardened version with full reproducibility and audit trail
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
import json
import yaml
import hashlib
import subprocess
from datetime import datetime
from sklearn.metrics import classification_report

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
from utils import (
    set_seed, get_dermamnist_dataloader, 
    calculate_metrics, save_results_csv, plot_confusion_matrix,
    verify_split_sizes, get_split_info
)

class PatchEmbedding(nn.Module):
    """Patch Embedding for Vision Transformer"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_patches=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Patch embedding: Conv2d or Linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # x: [B, C, H, W]
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
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer Block with Attention and MLP"""
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

class ViT(nn.Module):
    """Vision Transformer for Classification"""
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=192,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_classes=7,
        num_patches=64
    ):
        super().__init__()
        self.num_patches = num_patches
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, num_patches)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Normalize
        x = self.norm(x)
        
        # Classification head (use cls_token)
        x = x[:, 0]
        x = self.head(x)
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device).squeeze().long()
        
        optimizer.zero_grad()
        outputs = model(data)
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
            outputs = model(data)
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
            outputs = model(data)
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_outputs, all_targets

def get_config_hash(config_dict):
    """Generate hash of configuration for checkpoint verification"""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def get_git_hash():
    """Get current git commit hash, or 'unknown' if not in git repo"""
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return git_hash
    except:
        return 'unknown'

def log_system_info(device):
    """Log comprehensive system and environment information"""
    print('\n' + '='*80)
    print('SYSTEM & ENVIRONMENT INFO')
    print('='*80)
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'Device: {torch.cuda.get_device_name(0)}')
        print(f'Device count: {torch.cuda.device_count()}')
    print(f'Using device: {device}')
    print(f'Timestamp: {datetime.now().isoformat()}')
    print(f'Git hash: {get_git_hash()}')
    print('='*80 + '\n')

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_run_metadata(config, results_dir):
    """Save comprehensive run metadata including config, environment, and results"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'config': config,
        'config_hash': get_config_hash(config),
    }
    
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'run.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='ViT Baseline Training for DermaMNIST')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to YAML config file (overrides other args)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--embed-dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Explicit path to checkpoint to resume from (default: None = train from scratch)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable normalization')
    
    args = parser.parse_args()
    
    # Store original command-line values (to detect if user explicitly set them)
    cli_epochs = args.epochs
    cli_batch_size = args.batch_size
    cli_lr = args.lr
    cli_weight_decay = args.weight_decay
    cli_seed = args.seed
    
    # Load config from YAML if provided
    if args.config:
        print(f'Loading configuration from: {args.config}')
        config = load_config(args.config)
        
        # Use config values as defaults, but CLI args take precedence if different from defaults
        parser_defaults = parser.parse_args([])  # Get default values
        
        # Only override with config if CLI arg is still at default value
        if args.batch_size == parser_defaults.batch_size:
            args.batch_size = config['training']['batch_size']
        if args.epochs == parser_defaults.epochs:
            args.epochs = config['training']['epochs']
        if args.lr == parser_defaults.lr:
            args.lr = config['optimizer']['lr']
        if args.weight_decay == parser_defaults.weight_decay:
            args.weight_decay = config['optimizer']['weight_decay']
        if args.seed == parser_defaults.seed:
            args.seed = config['training']['seed']
        if args.img_size == parser_defaults.img_size:
            args.img_size = config['dataset']['img_size']
        if args.patch_size == parser_defaults.patch_size:
            args.patch_size = config['model']['patch_size']
        if args.embed_dim == parser_defaults.embed_dim:
            args.embed_dim = config['model']['embed_dim']
        if args.depth == parser_defaults.depth:
            args.depth = config['model']['depth']
        if args.num_heads == parser_defaults.num_heads:
            args.num_heads = config['model']['num_heads']
        if args.num_classes == parser_defaults.num_classes:
            args.num_classes = config['dataset']['num_classes']
    else:
        config = {}
    
    # Create config dict with final resolved values for metadata
    config = {
        'training': {'batch_size': args.batch_size, 'epochs': args.epochs, 'seed': args.seed},
        'optimizer': {'lr': args.lr, 'weight_decay': args.weight_decay},
        'dataset': {'img_size': args.img_size, 'num_classes': args.num_classes},
        'model': {'patch_size': args.patch_size, 'embed_dim': args.embed_dim, 
                 'depth': args.depth, 'num_heads': args.num_heads}
    }
    
    # Set seed for reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Log system information
    log_system_info(device)
    
    # Create directories
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save run metadata
    print('Saving run metadata...')
    save_run_metadata(config, args.results_dir)
    print(f'Config hash: {get_config_hash(config)}\n')
    
    # Verify dataset splits
    print('Verifying dataset splits...')
    try:
        verify_split_sizes(expected_train=7007, expected_val=1003, expected_test=2005)
        split_info = get_split_info()
        print(f'✓ Split sizes verified:')
        print(f'  Train: {split_info["train"]}')
        print(f'  Val:   {split_info["val"]}')
        print(f'  Test:  {split_info["test"]}\n')
    except ValueError as e:
        print(f'✗ SPLIT VERIFICATION FAILED: {e}')
        print('Aborting training due to split size mismatch.')
        sys.exit(1)
    
    # Data loaders with augmentation control
    use_augment = not args.no_augment
    use_normalize = not args.no_normalize
    
    print(f'Data augmentation: {"enabled" if use_augment else "disabled"}')
    print(f'Normalization: {"enabled" if use_normalize else "disabled"}\n')
    
    train_loader = utils.get_dermamnist_dataloader(
        'train', args.batch_size, download=True, size=args.img_size,
        basic_augment=use_augment, normalize=use_normalize
    )
    val_loader = utils.get_dermamnist_dataloader(
        'val', args.batch_size, download=False, size=args.img_size,
        basic_augment=False, normalize=use_normalize
    )
    test_loader = utils.get_dermamnist_dataloader(
        'test', args.batch_size, download=False, size=args.img_size,
        basic_augment=False, normalize=use_normalize
    )
    
    # Model
    num_patches = (args.img_size // args.patch_size) ** 2
    model = ViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        num_patches=num_patches
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
    
    # Resume from checkpoint ONLY if explicitly specified
    start_epoch = 1
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            print(f"✗ ERROR: Resume checkpoint not found: {args.resume_from}")
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print(f"RESUMING FROM CHECKPOINT: {args.resume_from}")
        print('='*80)
        checkpoint = torch.load(args.resume_from)
        
        # Verify config hash if available
        checkpoint_config_hash = checkpoint.get('config_hash', None)
        current_config_hash = get_config_hash(config)
        if checkpoint_config_hash and checkpoint_config_hash != current_config_hash:
            print(f"⚠ WARNING: Config hash mismatch!")
            print(f"  Checkpoint config hash: {checkpoint_config_hash}")
            print(f"  Current config hash:    {current_config_hash}")
            print(f"  This may indicate different hyperparameters.")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborting.")
                sys.exit(1)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])
        print(f"✓ Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        print('='*80 + '\n')
    else:
        print("\n" + "="*80)
        print("TRAINING FROM SCRATCH (no checkpoint specified)")
        print("="*80 + "\n")
    
    # Training
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_outputs, val_targets = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Scheduler step
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model with comprehensive metadata
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.ckpt_dir, 'best.ckpt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'config_hash': get_config_hash(config),
                'timestamp': datetime.now().isoformat(),
                'git_hash': get_git_hash(),
            }, checkpoint_path)
            
            # Save companion metadata JSON
            metadata = {
                'epoch': epoch,
                'val_acc': float(val_acc),
                'timestamp': datetime.now().isoformat(),
                'config_hash': get_config_hash(config),
                'git_hash': get_git_hash(),
            }
            metadata_path = os.path.join(args.ckpt_dir, 'best_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f'✓ New best model saved at epoch {epoch} (val_acc: {val_acc:.2f}%)')
        
        print(f'Epoch {epoch}/{args.epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    writer.close()
    
    # Test on best model
    print('\n' + '='*80)
    print('FINAL EVALUATION ON TEST SET')
    print('='*80)
    
    checkpoint = torch.load(os.path.join(args.ckpt_dir, 'best.ckpt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 0.0):.2f}%\n")
    
    test_outputs, test_targets = test(model, test_loader, criterion, device)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(test_outputs, test_targets, n_classes=args.num_classes)
    
    print(f'\n{"="*80}')
    print('TEST SET RESULTS (Final)')
    print('='*80)
    print(f'Accuracy:          {metrics["accuracy"]:.4f}')
    print(f'Precision (Macro): {metrics["precision_macro"]:.4f}')
    print(f'Recall (Macro):    {metrics["recall_macro"]:.4f}')
    print(f'F1 (Macro):        {metrics["f1_macro"]:.4f}')
    print('='*80 + '\n')
    
    # Save overall metrics summary
    save_results_csv(metrics, os.path.join(args.results_dir, 'metrics_summary.csv'), 'ViT')
    print(f'✓ Saved: {args.results_dir}/metrics_summary.csv')
    
    # Save per-class metrics separately
    import csv
    per_class_path = os.path.join(args.results_dir, 'per_class_metrics.csv')
    with open(per_class_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1', 'Support'])
        for i in range(args.num_classes):
            writer.writerow([
                f'Class_{i}',
                f'{metrics["precision_per_class"][i]:.4f}',
                f'{metrics["recall_per_class"][i]:.4f}',
                f'{metrics["f1_per_class"][i]:.4f}',
                int(metrics["support"][i])
            ])
    print(f'✓ Saved: {per_class_path}')
    
    # Generate and save classification report
    predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
    targets_np = test_targets.cpu().numpy()
    class_names = [f'Class_{i}' for i in range(args.num_classes)]
    report = classification_report(targets_np, predictions, target_names=class_names, digits=4)
    
    report_path = os.path.join(args.results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write('Classification Report - ViT Baseline on DermaMNIST Test Set\n')
        f.write('='*80 + '\n\n')
        f.write(report)
        f.write('\n\n')
        f.write(f'Timestamp: {datetime.now().isoformat()}\n')
        f.write(f'Config hash: {get_config_hash(config)}\n')
        f.write(f'Best epoch: {checkpoint.get("epoch", "unknown")}\n')
    print(f'✓ Saved: {report_path}')
    
    # Save confusion matrix
    plot_confusion_matrix(metrics, os.path.join(args.results_dir, 'confusion_matrix.png'))
    print(f'✓ Saved: {args.results_dir}/confusion_matrix.png')
    
    # Save training history
    history_path = os.path.join(args.results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_epoch': int(checkpoint.get('epoch', 0)),
            'best_val_acc': float(best_val_acc),
        }, f, indent=2)
    print(f'✓ Saved: {history_path}')
    
    print(f'\n{"="*80}')
    print('ALL RESULTS SAVED SUCCESSFULLY')
    print(f'Results directory: {args.results_dir}/')
    print('='*80)

if __name__ == '__main__':
    main()

