#!/usr/bin/env python
"""
Quick syntax and import checker for AutoMAE DermaMNIST
"""

import sys
import os

def check_file(filepath):
    """Check if a Python file has syntax errors"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        print(f"✓ {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: {e}")
        return False
    except Exception as e:
        print(f"? {filepath}: {e}")
        return False

def main():
    print("="*80)
    print("Checking Python Syntax")
    print("="*80)
    
    files_to_check = [
        'datasets/derma_dataset.py',
        'models_mae_derma.py',
        'train_automae_derma.py',
        'eval_classification_derma.py',
        'engine_pretrain_derma.py',
    ]
    
    all_ok = True
    for file in files_to_check:
        if os.path.exists(file):
            if not check_file(file):
                all_ok = False
        else:
            print(f"✗ {file}: File not found")
            all_ok = False
    
    print("\n" + "="*80)
    if all_ok:
        print("✓ All syntax checks passed!")
        print("="*80)
        
        # Try importing key modules
        print("\nChecking imports...")
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import models_mae_derma
            print("✓ models_mae_derma imported")
            from datasets import derma_dataset
            print("✓ datasets.derma_dataset imported")
            print("\n✓ All imports successful!")
        except Exception as e:
            print(f"✗ Import error: {e}")
            all_ok = False
    else:
        print("✗ Some checks failed!")
        print("="*80)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

