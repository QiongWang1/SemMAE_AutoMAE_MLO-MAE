"""
ViT Baseline for PathMNIST Classification (independent from DermaMNIST files)
"""
import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils_pathmnist import (
    set_seed,
    get_pathmnist_dataloader,
    verify_split_sizes,
    get_split_info,
    calculate_metrics,
    save_json,
)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
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
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=192,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 num_classes=9):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
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
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for data, targets in pbar:
        data = data.to(device)
        targets = targets.to(device).squeeze().long()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device, desc="[Val]"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc)
        for data, targets in pbar:
            data = data.to(device)
            targets = targets.to(device).squeeze().long()
            outputs = model(data)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_outputs.append(outputs)
            all_targets.append(targets)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
    loss = running_loss / len(loader)
    acc = 100. * correct / total
    return loss, acc, torch.cat(all_outputs, dim=0), torch.cat(all_targets, dim=0)


def main():
    parser = argparse.ArgumentParser(description="PathMNIST ViT Baseline")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    # Output directories (will be absolute paths under job/<JOB_ID>/)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--events-dir", type=str, required=True)
    parser.add_argument("--preds-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.events_dir, exist_ok=True)
    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("PathMNIST ViT Baseline")
    print("=" * 80)
    print(f"Device: {device}")

    print("Verifying dataset splits...")
    verify_split_sizes()
    split_info = get_split_info()
    print(f"\u2713 Split sizes: train={split_info['train']} val={split_info['val']} test={split_info['test']}")

    use_augment = not args.no_augment
    use_normalize = not args.no_normalize
    print(f"Data augmentation: {'enabled' if use_augment else 'disabled'}")
    print(f"Normalization: {'enabled' if use_normalize else 'disabled'}")

    train_loader = get_pathmnist_dataloader("train", args.batch_size, size=args.img_size, download=True, basic_augment=use_augment, normalize=use_normalize)
    val_loader = get_pathmnist_dataloader("val", args.batch_size, size=args.img_size, download=False, basic_augment=False, normalize=use_normalize)
    test_loader = get_pathmnist_dataloader("test", args.batch_size, size=args.img_size, download=False, basic_augment=False, normalize=use_normalize)

    model = ViT(img_size=args.img_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                num_classes=args.num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=os.path.join(args.events_dir, "tensorboard"))

    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="[Val]")
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.ckpt_dir, "best_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": float(val_acc),
                "timestamp": datetime.now().isoformat(),
            }, ckpt_path)
            print(f"\u2713 Saved best checkpoint: {ckpt_path}")

        print(f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")

    writer.close()

    # Load best for final test
    best_ckpt = torch.load(os.path.join(args.ckpt_dir, "best_checkpoint.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt.get('epoch')}, val_acc={best_ckpt.get('val_acc'):.2f}%")

    test_loss, test_acc, test_logits, test_targets = evaluate(model, test_loader, criterion, device, desc="[Test]")

    # Save predictions and logits
    preds = torch.argmax(test_logits, dim=1).cpu().numpy()
    npy_logits_path = os.path.join(args.preds_dir, "test_logits.npy")
    npy_preds_path = os.path.join(args.preds_dir, "test_preds.npy")
    torch.save(test_logits.cpu(), os.path.join(args.preds_dir, "test_logits.pt"))
    import numpy as np
    np.save(npy_logits_path, test_logits.cpu().numpy())
    np.save(npy_preds_path, preds)

    # Metrics summary
    metrics = calculate_metrics(test_logits, test_targets, n_classes=args.num_classes)
    metrics.update({
        "test_loss": float(test_loss),
        "test_acc_percent": float(test_acc),
        "best_val_acc_percent": float(best_val_acc),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
    })

    # Save structured results
    save_json(metrics, os.path.join(args.results_dir, "metrics.json"))

    # Also write compact CSV-like summaries
    with open(os.path.join(args.results_dir, "metrics_summary.csv"), "w") as f:
        f.write("metric,value\n")
        f.write(f"accuracy,{metrics['accuracy']:.6f}\n")
        f.write(f"precision_macro,{metrics['precision_macro']:.6f}\n")
        f.write(f"recall_macro,{metrics['recall_macro']:.6f}\n")
        f.write(f"f1_macro,{metrics['f1_macro']:.6f}\n")

    # Per-class metrics
    with open(os.path.join(args.results_dir, "per_class_metrics.csv"), "w") as f:
        f.write("class,precision,recall,f1,support\n")
        for i in range(args.num_classes):
            f.write(
                f"{i},{metrics['precision_per_class'][i]:.6f},{metrics['recall_per_class'][i]:.6f},{metrics['f1_per_class'][i]:.6f},{metrics['support'][i]}\n"
            )

    # Minimal console summary
    print("\n" + "=" * 80)
    print("TEST SET RESULTS (PathMNIST)")
    print("=" * 80)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 (Macro):        {metrics['f1_macro']:.4f}")
    print("=" * 80)
    print(f"Saved results to: {args.results_dir}")
    print(f"Saved predictions to: {args.preds_dir}")


if __name__ == "__main__":
    main()


