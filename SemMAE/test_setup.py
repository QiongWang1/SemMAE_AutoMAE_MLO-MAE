"""
Dry-run test to verify SemMAE setup for DermaMNIST
Tests: dataset loading, model instantiation, forward/backward pass
"""
import sys
import torch
import torch.nn as nn
from torchvision import transforms

print("="*80)
print("SemMAE DermaMNIST Setup Verification")
print("="*80)

# Test 1: Import checks
print("\n[1/6] Testing imports...")
try:
    import timm
    print(f"  ✓ timm version: {timm.__version__}")
    # Compatible with timm 0.9.x and later
except Exception as e:
    print(f"  ✗ timm import failed: {e}")
    sys.exit(1)

try:
    from medmnist import DermaMNIST
    print(f"  ✓ medmnist imported successfully")
except Exception as e:
    print(f"  ✗ medmnist import failed: {e}")
    sys.exit(1)

try:
    import util.misc as misc
    import util.lr_sched as lr_sched
    import util.pos_embed as pos_embed
    print(f"  ✓ util modules imported successfully")
except Exception as e:
    print(f"  ✗ util import failed: {e}")
    sys.exit(1)

try:
    import models_mae_derma as models_mae
    print(f"  ✓ models_mae_derma imported successfully")
except Exception as e:
    print(f"  ✗ models_mae_derma import failed: {e}")
    sys.exit(1)

try:
    from derma_dataloader import build_derma_dataset
    print(f"  ✓ derma_dataloader imported successfully")
except Exception as e:
    print(f"  ✗ derma_dataloader import failed: {e}")
    sys.exit(1)

# Test 2: CUDA availability
print("\n[2/6] Testing CUDA...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    device = torch.device('cuda')
else:
    print(f"  ⚠ CUDA not available, using CPU")
    device = torch.device('cpu')

# Test 3: Dataset loading
print("\n[3/6] Testing dataset loading...")
try:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = build_derma_dataset(split='train', download=True, transform=transform, target_size=32)
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    
    # Test a single sample
    img, label = dataset[0]
    print(f"  ✓ Sample shape: {img.shape}, Label: {label}")
    assert img.shape == (3, 32, 32), f"Expected (3, 32, 32), got {img.shape}"
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_img, batch_label = next(iter(dataloader))
    print(f"  ✓ Batch shape: {batch_img.shape}, Labels: {batch_label.shape}")
    
except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model instantiation
print("\n[4/6] Testing model instantiation...")
try:
    model = models_mae.__dict__['mae_vit_small'](norm_pix_loss=True)
    model.to(device)
    print(f"  ✓ Model created successfully")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {num_params:,}")
    
except Exception as e:
    print(f"  ✗ Model instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\n[5/6] Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(2, 3, 32, 32).to(device)
        loss, pred, mask = model(test_input, mask_ratio=0.75)
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  ✓ Prediction shape: {pred.shape}")
        print(f"  ✓ Mask shape: {mask.shape}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Backward pass (training mode)
print("\n[6/6] Testing backward pass...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Single training step
    batch_img = batch_img.to(device)
    optimizer.zero_grad()
    loss, pred, mask = model(batch_img, mask_ratio=0.75)
    loss.backward()
    optimizer.step()
    
    print(f"  ✓ Backward pass successful")
    print(f"  ✓ Training step loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"  ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("✓ All tests passed successfully!")
print("="*80)
print("\nYou can now submit the training job:")
print("  sbatch train_semmae.sh")
print("\nOr run training directly:")
print("  python train_semmae_derma.py --epochs 200 --batch_size 128")
print("="*80)

