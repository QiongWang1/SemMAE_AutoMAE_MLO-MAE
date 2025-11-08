"""
Visualize MAE reconstructions on DermaMNIST
Shows original, masked, and reconstructed images
All outputs saved to DermaMNIST/Output/visualizations_<timestamp>/
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create output directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f'DermaMNIST/Output/visualizations_{timestamp}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print()

from medmnist import DermaMNIST, INFO
from torch.utils.data import DataLoader
import cifar_mae_model as models_mae

print("=" * 80)
print("MLO-MAE DermaMNIST - Reconstruction Visualization")
print("=" * 80)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data normalization
DERMA_MEAN = [0.485, 0.456, 0.406]
DERMA_STD = [0.229, 0.224, 0.225]

# Load test dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(DERMA_MEAN, DERMA_STD),
])

test_dataset = DermaMNIST(split='test', transform=test_transform, download=True, root='./data')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)

# Load pretrained MAE model
print("Loading pretrained MAE model...")
model = models_mae.cifar10_mae_vit_base_patch16_dec512d8b(norm_pix_loss=True)
model = model.to(device)

checkpoint_path = './checkpoint_dermamnist/pretrain.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"Warning: Checkpoint not found at {checkpoint_path}")
    print("Proceeding with random weights for demonstration")

model.eval()

# Get a batch of images
images, labels = next(iter(test_loader))
images = images.to(device)

# Generate reconstructions
print("Generating reconstructions...")
with torch.no_grad():
    # Forward pass through encoder
    latent, mask, ids_restore = model.forward_encoder(images, mask_ratio=0.75)
    # Reconstruct
    pred = model.forward_decoder(latent, ids_restore)

# Unnormalize function
def unnormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)

# Convert to numpy
images_np = images.cpu()
pred_np = pred.cpu()
mask_np = mask.cpu()

# Create visualization
n_samples = min(8, images.shape[0])
fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))

if n_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(n_samples):
    # Original image
    img_orig = unnormalize(images_np[i], DERMA_MEAN, DERMA_STD)
    img_orig = img_orig.permute(1, 2, 0).numpy()
    
    # Masked image
    img_masked = model.unpatchify(pred_np[i:i+1]).squeeze(0)
    img_masked = unnormalize(img_masked, DERMA_MEAN, DERMA_STD)
    
    # Apply mask for visualization
    mask_img = mask_np[i].unsqueeze(-1).unsqueeze(-1)
    mask_img = mask_img.repeat(1, model.patch_embed.patch_size[0], model.patch_embed.patch_size[0])
    mask_img = mask_img.reshape(int(mask_np.shape[1]**0.5), int(mask_np.shape[1]**0.5), 
                                 model.patch_embed.patch_size[0], model.patch_embed.patch_size[0])
    mask_img = mask_img.permute(0, 2, 1, 3).reshape(32, 32).numpy()
    
    img_masked_vis = img_orig.copy()
    img_masked_vis[mask_img > 0.5] = 0.5  # Gray out masked regions
    
    # Reconstructed image
    img_recon = img_masked.permute(1, 2, 0).numpy()
    
    # Plot
    axes[i, 0].imshow(img_orig)
    axes[i, 0].set_title(f'Original (Label: {labels[i].item()})', fontsize=10)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(img_masked_vis)
    axes[i, 1].set_title('Masked (75%)', fontsize=10)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(img_recon)
    axes[i, 2].set_title('Reconstructed', fontsize=10)
    axes[i, 2].axis('off')

plt.suptitle('MLO-MAE Reconstructions on DermaMNIST', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/reconstruction_samples.png', dpi=300, bbox_inches='tight')
print(f"Reconstruction visualization saved to {OUTPUT_DIR}/reconstruction_samples.png")

# Close plot
plt.close()

# Create summary file
summary = f"""# MAE Reconstruction Visualization

**Timestamp**: {timestamp}
**Model**: {checkpoint_path}
**Samples visualized**: {n_samples}

## Files Generated
- reconstruction_samples.png - MAE reconstruction examples

## Model Info
- Encoder: pretrain.pth
- Mask ratio: 75%
- Patch size: 2×2
- Image size: 32×32

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{OUTPUT_DIR}/README.md', 'w') as f:
    f.write(summary)

print("=" * 80)
print("✓ Visualization complete!")
print(f"✓ All outputs saved to: {OUTPUT_DIR}/")
print("=" * 80)

