#!/usr/bin/env python3
"""
Parse hyperparameters from DermaMNIST log files
Extracts pretrain and finetune parameters to use as defaults for PathMNIST
"""

import os
import re
import json


def parse_pretrain_params(log_file):
    """Parse pretrain parameters from DermaMNIST pretrain log"""
    params = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find Namespace(...) block
    if 'Namespace(' in content:
        namespace_start = content.find('Namespace(')
        namespace_end = content.find(')', namespace_start)
        namespace_str = content[namespace_start:namespace_end+1]
        
        # Extract parameters
        params['batch_size'] = int(re.search(r'batch_size=(\d+)', namespace_str).group(1))
        params['epochs'] = int(re.search(r'epochs=(\d+)', namespace_str).group(1))
        params['model'] = re.search(r"model='([^']+)'", namespace_str).group(1)
        params['input_size'] = int(re.search(r'input_size=(\d+)', namespace_str).group(1))
        params['mask_ratio'] = float(re.search(r'mask_ratio=([\d.]+)', namespace_str).group(1))
        params['norm_pix_loss'] = 'norm_pix_loss=True' in namespace_str
        params['weight_decay'] = float(re.search(r'weight_decay=([\d.e-]+)', namespace_str).group(1))
        params['lr'] = float(re.search(r'lr=([\d.e-]+)', namespace_str).group(1))
        params['blr'] = float(re.search(r'blr=([\d.e-]+)', namespace_str).group(1))
        params['min_lr'] = float(re.search(r'min_lr=([\d.e-]+)', namespace_str).group(1))
        params['warmup_epochs'] = int(re.search(r'warmup_epochs=(\d+)', namespace_str).group(1))
        params['num_workers'] = int(re.search(r'num_workers=(\d+)', namespace_str).group(1))
        params['finetune_lr'] = float(re.search(r'finetune_lr=([\d.e-]+)', namespace_str).group(1))
        params['finetune_batchsize'] = int(re.search(r'finetune_batchsize=(\d+)', namespace_str).group(1))
        params['finetune_weight_decay'] = float(re.search(r'finetune_weight_decay=([\d.e-]+)', namespace_str).group(1))
        params['masking_lr'] = float(re.search(r'masking_lr=([\d.e-]+)', namespace_str).group(1))
        params['masking_batchsize'] = int(re.search(r'masking_batchsize=(\d+)', namespace_str).group(1))
        params['masking_weight_decay'] = float(re.search(r'masking_weight_decay=([\d.e-]+)', namespace_str).group(1))
    
    return params


def parse_finetune_params(log_file):
    """Parse finetune parameters from DermaMNIST finetune log"""
    params = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find Namespace(...) block
    if 'Namespace(' in content:
        namespace_start = content.find('Namespace(')
        namespace_end = content.find(')', namespace_start)
        namespace_str = content[namespace_start:namespace_end+1]
        
        # Extract parameters
        params['lr'] = float(re.search(r'lr=([\d.e-]+)', namespace_str).group(1))
        params['minlr'] = float(re.search(r'minlr=([\d.e-]+)', namespace_str).group(1))
        params['weight_decay'] = float(re.search(r'weight_decay=([\d.e-]+)', namespace_str).group(1))
        params['opt'] = re.search(r"opt='([^']+)'", namespace_str).group(1)
        params['net'] = re.search(r"net='([^']+)'", namespace_str).group(1)
        
        bs_match = re.search(r"bs='(\d+)'", namespace_str)
        if bs_match:
            params['batch_size'] = int(bs_match.group(1))
        
        size_match = re.search(r"size='(\d+)'", namespace_str)
        if size_match:
            params['input_size'] = int(size_match.group(1))
        
        params['n_epochs'] = int(re.search(r'n_epochs=(\d+)', namespace_str).group(1))
        params['num_workers'] = int(re.search(r'num_workers=(\d+)', namespace_str).group(1))
        
        patch_match = re.search(r'patch=(\d+)', namespace_str)
        if patch_match:
            params['patch'] = int(patch_match.group(1))
    
    return params


if __name__ == '__main__':
    # Paths to DermaMNIST log files
    pretrain_log = '/projects/weilab/qiongwang/MLO_MAE/DermaMNIST/job/job_pretrain/mlo_mae_dermamnist_pretrain_1792116.out'
    finetune_log = '/projects/weilab/qiongwang/MLO_MAE/DermaMNIST/job/job_finetune/mlo_mae_dermamnist_finetune_epoch1000_1798444.out'
    
    print("=" * 80)
    print("Extracting Hyperparameters from DermaMNIST Logs")
    print("=" * 80)
    
    # Parse pretrain params
    print("\n📋 PRETRAIN PARAMETERS")
    print("-" * 80)
    pretrain_params = parse_pretrain_params(pretrain_log)
    for key, value in sorted(pretrain_params.items()):
        print(f"  {key:30s}: {value}")
    
    # Parse finetune params
    print("\n📋 FINETUNE PARAMETERS")
    print("-" * 80)
    finetune_params = parse_finetune_params(finetune_log)
    for key, value in sorted(finetune_params.items()):
        print(f"  {key:30s}: {value}")
    
    print("\n" + "=" * 80)
    
    # Save to JSON for easy reference
    output_file = '/projects/weilab/qiongwang/MLO_MAE/PathMNIST/configs/derma_params.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_params = {
        'pretrain': pretrain_params,
        'finetune': finetune_params
    }
    
    with open(output_file, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\n✓ Parameters saved to: {output_file}")
    print("=" * 80)


