"""
Verify that all files are in place and ready for training
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def main():
    print("=" * 60)
    print("Baseline Setup Verification")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent
    issues = []
    
    # Check core files
    print("Core Files:")
    core_files = [
        (base_dir / "utils.py", "Utilities module"),
        (base_dir / "generate_comparison.py", "Comparison script"),
        (base_dir / "README.md", "Main README"),
        (base_dir / "requirements.txt", "Requirements file"),
        (base_dir / "submit_all.sh", "Submit all script"),
    ]
    
    for filepath, desc in core_files:
        if not check_file_exists(filepath, desc):
            issues.append(f"Missing: {filepath}")
    
    print()
    
    # Check ViT
    print("ViT Model:")
    vit_files = [
        (base_dir / "ViT" / "train_vit.py", "ViT training script"),
        (base_dir / "ViT" / "train_vit.sh", "ViT SLURM script"),
    ]
    
    for filepath, desc in vit_files:
        if not check_file_exists(filepath, desc):
            issues.append(f"Missing: {filepath}")
    
    print()
    
    # Check MAE
    print("MAE Model:")
    mae_files = [
        (base_dir / "MAE" / "train_mae.py", "MAE training script"),
        (base_dir / "MAE" / "train_mae.sh", "MAE SLURM script"),
    ]
    
    for filepath, desc in mae_files:
        if not check_file_exists(filepath, desc):
            issues.append(f"Missing: {filepath}")
    
    print()
    
    # Check U-MAE
    print("U-MAE Model:")
    umae_files = [
        (base_dir / "U-MAE" / "train_umae.py", "U-MAE training script"),
        (base_dir / "U-MAE" / "train_umae.sh", "U-MAE SLURM script"),
    ]
    
    for filepath, desc in umae_files:
        if not check_file_exists(filepath, desc):
            issues.append(f"Missing: {filepath}")
    
    print()
    print("=" * 60)
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("Please fix the above issues before proceeding.")
        sys.exit(1)
    else:
        print("✓ All files are in place!")
        print()
        print("Next steps:")
        print("1. Submit jobs: bash submit_all.sh")
        print("2. Or submit individually: sbatch ViT/train_vit.sh")
        print("3. Monitor: squeue -u $USER")
        print("4. After completion: python generate_comparison.py")
        print()
        print("Ready to start training!")
    
    print("=" * 60)

if __name__ == '__main__':
    main()

