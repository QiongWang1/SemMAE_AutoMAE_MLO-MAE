"""
Test to verify the distributed initialization fix
Simulates the SLURM environment and tests the fixed init_distributed_mode
"""
import sys
import os
import torch

print("="*80)
print("Distributed Initialization Fix - Verification Test")
print("="*80)

# Simulate SLURM single-GPU environment
os.environ['SLURM_PROCID'] = '0'
os.environ['SLURM_NTASKS'] = '1'

print("\n[1/4] Testing imports...")
try:
    import util.misc as misc
    print("  ✓ util.misc imported")
    
    import models_mae_derma as models_mae
    print("  ✓ models_mae_derma imported")
    
    from derma_dataloader import build_derma_dataset
    print("  ✓ derma_dataloader imported")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("\n[2/4] Testing distributed mode initialization (single GPU)...")
try:
    import argparse
    
    # Create minimal args like the training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_on_itp', action='store_true', default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args([])
    
    # This should NOT throw an error now
    misc.init_distributed_mode(args)
    
    print(f"  ✓ init_distributed_mode completed")
    print(f"  ✓ args.distributed = {args.distributed}")
    print(f"  ✓ args.rank = {args.rank}")
    print(f"  ✓ args.world_size = {args.world_size}")
    print(f"  ✓ args.gpu = {args.gpu}")
    
    if args.distributed:
        print("  ✗ ERROR: Should NOT be in distributed mode with single GPU!")
        sys.exit(1)
    else:
        print("  ✓ Correctly identified as non-distributed mode")
    
except Exception as e:
    print(f"  ✗ Distributed init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/4] Testing CUDA device setup...")
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ Current device: {torch.cuda.current_device()}")
    else:
        device = torch.device('cpu')
        print(f"  ⚠ CUDA not available, using CPU")
except Exception as e:
    print(f"  ✗ CUDA setup failed: {e}")
    sys.exit(1)

print("\n[4/4] Testing world_size and rank functions...")
try:
    world_size = misc.get_world_size()
    rank = misc.get_rank()
    is_main = misc.is_main_process()
    
    print(f"  ✓ World size: {world_size}")
    print(f"  ✓ Rank: {rank}")
    print(f"  ✓ Is main process: {is_main}")
    
    if world_size != 1:
        print(f"  ✗ ERROR: Expected world_size=1, got {world_size}")
        sys.exit(1)
    if rank != 0:
        print(f"  ✗ ERROR: Expected rank=0, got {rank}")
        sys.exit(1)
    if not is_main:
        print(f"  ✗ ERROR: Expected is_main_process=True")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ World size/rank test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ All tests passed! Distributed initialization fix is working.")
print("="*80)
print("\nThe fix correctly handles single-GPU SLURM jobs:")
print("  • Detects SLURM_NTASKS=1")
print("  • Skips distributed initialization")
print("  • Sets args.distributed=False")
print("  • Configures single GPU properly")
print("\nReady to submit job:")
print("  cd /projects/weilab/qiongwang/baseline_method/SemMAE")
print("  sbatch train_semmae.sh")
print("="*80)

