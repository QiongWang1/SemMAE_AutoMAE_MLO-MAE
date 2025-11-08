#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly for PathMNIST scripts
Run from MLO_MAE root: python PathMNIST/test_imports.py
"""

import sys
import os

# Add MLO_MAE root to path (same as main_pathmnist.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
mlo_mae_root = os.path.join(script_dir, '..')
sys.path.insert(0, mlo_mae_root)

print("Testing PathMNIST import configuration...")
print(f"Script directory: {script_dir}")
print(f"MLO_MAE root: {mlo_mae_root}")
print(f"sys.path[0]: {sys.path[0]}")
print("")

# Test all critical imports
try:
    print("✓ Importing torch...")
    import torch
    
    print("✓ Importing util.misc...")
    import util.misc as misc
    
    print("✓ Importing util.lr_sched...")
    import util.lr_sched as lr_sched
    
    print("✓ Importing cifar_mae_model...")
    import cifar_mae_model as models_mae_mlo_ddp
    
    print("✓ Importing cifar_vit_mlo...")
    import cifar_vit_mlo as models_vit_mlo_ddp
    
    print("✓ Importing betty...")
    from betty.engine import Engine
    from betty.problems import ImplicitProblem
    
    print("✓ Importing medmnist...")
    from medmnist import PathMNIST, INFO
    
    print("✓ Importing util.pos_embed...")
    from util.pos_embed import get_2d_sincos_pos_embed
    
    print("")
    print("=" * 60)
    print("SUCCESS! All imports working correctly.")
    print("=" * 60)
    print("")
    print("PathMNIST is ready for training.")
    print("")
    print("Next steps:")
    print("  1. Submit pretrain job:")
    print("     cd /projects/weilab/qiongwang/MLO_MAE")
    print("     sbatch PathMNIST/job/job_pretrain/job_pretrain_pathmnist.sbatch")
    print("")
    print("  2. Monitor job:")
    print("     tail -f mlo_mae_pathmnist_pretrain_*.out")
    
except ImportError as e:
    print("")
    print("=" * 60)
    print("FAILED! Import error:")
    print("=" * 60)
    print(f"Error: {e}")
    print("")
    print("This indicates the sys.path fix did not work correctly.")
    sys.exit(1)

