"""
Quick test to verify the training setup works with a single batch
This catches errors before submitting a long job
"""
import sys
import torch
from torchvision import transforms

print("="*80)
print("SemMAE Single Batch Test")
print("="*80)

# Test imports
print("\n[1/5] Testing imports...")
try:
    import timm
    print(f"  ✓ timm version: {timm.__version__}")
    
    from medmnist import DermaMNIST
    print(f"  ✓ medmnist imported")
    
    import util.misc as misc
    print(f"  ✓ util.misc imported")
    
    import models_mae_derma as models_mae
    print(f"  ✓ models_mae_derma imported")
    
    from derma_dataloader import build_derma_dataset
    print(f"  ✓ derma_dataloader imported")
    
    from train_semmae_derma import add_weight_decay
    print(f"  ✓ add_weight_decay function imported")
    
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test CUDA
print("\n[2/5] Testing CUDA...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print(f"  ⚠ CUDA not available, using CPU")
    device = torch.device('cpu')

# Test dataset
print("\n[3/5] Testing dataset...")
try:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = build_derma_dataset(split='train', download=True, transform=transform, target_size=32)
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_img, batch_label = next(iter(dataloader))
    print(f"  ✓ Batch shape: {batch_img.shape}")
    
except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model
print("\n[4/5] Testing model...")
try:
    model = models_mae.__dict__['mae_vit_small'](norm_pix_loss=True)
    model.to(device)
    print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test weight decay function
    param_groups = add_weight_decay(model, weight_decay=0.05)
    print(f"  ✓ Weight decay groups: {len(param_groups)} groups")
    
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, betas=(0.9, 0.95))
    print(f"  ✓ Optimizer created")
    
except Exception as e:
    print(f"  ✗ Model setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training step
print("\n[5/5] Testing training step...")
try:
    model.train()
    batch_img = batch_img.to(device)
    
    optimizer.zero_grad()
    loss, pred, mask = model(batch_img, mask_ratio=0.75)
    loss.backward()
    optimizer.step()
    
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Backward pass successful")
    
except Exception as e:
    print(f"  ✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ All tests passed! Ready to submit job.")
print("="*80)
print("\nTo submit the job:")
print("  cd /projects/weilab/qiongwang/baseline_method/SemMAE")
print("  sbatch train_semmae.sh")
print("="*80)

