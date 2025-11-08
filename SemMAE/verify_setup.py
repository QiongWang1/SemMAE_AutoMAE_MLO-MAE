#!/usr/bin/env python3
"""
Verification script for PathMNIST SemMAE setup
Run this to verify all dependencies and files are in place
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ✗ Python 3.7+ required")
        return False
    return True

def check_imports():
    """Check required packages"""
    print("\nChecking required packages...")
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'Timm',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'medmnist': 'MedMNIST',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                print(f"  ✓ {name} {sklearn.__version__}")
            elif package == 'PIL':
                import PIL
                print(f"  ✓ {name} {PIL.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✓ {name} {version}")
        except ImportError:
            print(f"  ✗ {name} not found")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Version: {torch.version.cuda}")
            print(f"    - Device count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"    - Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available (CPU only)")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False
    return True

def check_files():
    """Check required files exist"""
    print("\nChecking required files...")
    files = [
        'pathmnist_dataloader.py',
        'train_semmae_pathmnist.py',
        'engine_pretrain_pathmnist.py',
        'evaluate_semmae_pathmnist.py',
        'job/train_semmae_pathmnist.sh',
        'README.md',
        'QUICKSTART.md',
        'SUBMISSION_GUIDE.md'
    ]
    
    base_dir = Path(__file__).parent
    all_ok = True
    for file in files:
        file_path = base_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            all_ok = False
    
    return all_ok

def check_parent_modules():
    """Check if parent SemMAE modules are accessible"""
    print("\nChecking parent SemMAE modules...")
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    modules = [
        'util.misc',
        'util.lr_sched',
        'models_mae_derma'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} not found: {e}")
            all_ok = False
    
    return all_ok

def check_dataset():
    """Check if dataset can be loaded"""
    print("\nChecking dataset access...")
    try:
        from pathmnist_dataloader import build_pathmnist_dataset
        print("  ✓ DataLoader module imported")
        
        # Try to build dataset (will download if needed)
        print("  - Attempting to load PathMNIST dataset...")
        dataset = build_pathmnist_dataset(split='train', download=True)
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")
        
        # Test getting one sample
        img, label = dataset[0]
        print(f"  ✓ Sample shape: {img.shape}, label: {label}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False

def main():
    print("="*70)
    print("PathMNIST SemMAE Setup Verification")
    print("="*70)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_imports()))
    results.append(("CUDA Support", check_cuda()))
    results.append(("Required Files", check_files()))
    results.append(("Parent Modules", check_parent_modules()))
    results.append(("Dataset Access", check_dataset()))
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    all_passed = True
    for name, status in results:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}: {'PASSED' if status else 'FAILED'}")
        if not status:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All checks passed! Setup is complete and ready to use.")
        print("\nNext step:")
        print("  sbatch job/train_semmae_pathmnist.sh")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  1. Activate conda environment:")
        print("     conda activate /projects/weilab/qiongwang/envs/baseline_derma")
        print("  2. Install missing packages:")
        print("     pip install torch torchvision timm medmnist scikit-learn")
        return 1

if __name__ == '__main__':
    sys.exit(main())






