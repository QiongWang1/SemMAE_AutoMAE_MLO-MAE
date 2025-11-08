#!/usr/bin/env python3
"""
Data Leakage Detector for DermaMNIST
Checks for any overlap between train/val/test splits
"""
import sys
import os
import csv
import hashlib
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from medmnist import DermaMNIST
from tqdm import tqdm

def get_image_hash(img_array):
    """Generate hash of image array for comparison"""
    return hashlib.md5(img_array.tobytes()).hexdigest()

def load_split_hashes(split_name):
    """Load all images from a split and compute their hashes"""
    print(f"Loading {split_name} split...")
    dataset = DermaMNIST(split=split_name, transform=None, download=False)
    
    hashes = []
    for i in tqdm(range(len(dataset)), desc=f"Hashing {split_name}", ncols=80):
        img, label = dataset[i]
        img_hash = get_image_hash(np.array(img))
        hashes.append({
            'split': split_name,
            'index': i,
            'hash': img_hash,
            'label': int(label[0]) if isinstance(label, np.ndarray) else int(label)
        })
    
    return hashes

def find_duplicates(all_hashes):
    """Find duplicate hashes across splits"""
    hash_to_samples = defaultdict(list)
    
    for sample in all_hashes:
        hash_to_samples[sample['hash']].append(sample)
    
    # Find hashes that appear in multiple locations
    duplicates = []
    for hash_val, samples in hash_to_samples.items():
        if len(samples) > 1:
            duplicates.append((hash_val, samples))
    
    return duplicates

def check_cross_split_leakage(split1_hashes, split2_hashes, name1, name2):
    """Check for leakage between two specific splits"""
    hash_set1 = set(h['hash'] for h in split1_hashes)
    hash_set2 = set(h['hash'] for h in split2_hashes)
    
    overlap = hash_set1 & hash_set2
    
    if overlap:
        print(f"  ✗ Found {len(overlap)} duplicate(s) between {name1} and {name2}")
        return overlap
    else:
        print(f"  ✓ No leakage between {name1} and {name2}")
        return set()

def main():
    """Run comprehensive leakage detection"""
    print("="*80)
    print("DATA LEAKAGE DETECTION FOR DERMAMNIST")
    print("="*80 + "\n")
    
    # Load all splits
    print("Step 1: Loading and hashing all splits...")
    print("-" * 80)
    
    train_hashes = load_split_hashes('train')
    val_hashes = load_split_hashes('val')
    test_hashes = load_split_hashes('test')
    
    print(f"\nLoaded:")
    print(f"  Train: {len(train_hashes)} samples")
    print(f"  Val:   {len(val_hashes)} samples")
    print(f"  Test:  {len(test_hashes)} samples")
    
    # Check for within-split duplicates
    print("\n" + "="*80)
    print("Step 2: Checking for within-split duplicates...")
    print("-" * 80)
    
    for split_name, split_hashes in [('train', train_hashes), 
                                      ('val', val_hashes), 
                                      ('test', test_hashes)]:
        hash_counts = defaultdict(int)
        for h in split_hashes:
            hash_counts[h['hash']] += 1
        
        duplicates = {k: v for k, v in hash_counts.items() if v > 1}
        if duplicates:
            print(f"  ✗ {split_name}: Found {len(duplicates)} duplicate hash(es)")
            print(f"     Total duplicate samples: {sum(duplicates.values()) - len(duplicates)}")
        else:
            print(f"  ✓ {split_name}: No within-split duplicates")
    
    # Check for cross-split leakage
    print("\n" + "="*80)
    print("Step 3: Checking for cross-split leakage...")
    print("-" * 80)
    
    train_val_overlap = check_cross_split_leakage(train_hashes, val_hashes, "train", "val")
    train_test_overlap = check_cross_split_leakage(train_hashes, test_hashes, "train", "test")
    val_test_overlap = check_cross_split_leakage(val_hashes, test_hashes, "val", "test")
    
    # Combine all hashes and find all duplicates
    print("\n" + "="*80)
    print("Step 4: Detailed duplicate analysis...")
    print("-" * 80)
    
    all_hashes = train_hashes + val_hashes + test_hashes
    all_duplicates = find_duplicates(all_hashes)
    
    # Generate leakage report
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'leakage_report.csv')
    
    print(f"\nGenerating report: {report_path}")
    
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Hash', 'Split1', 'Index1', 'Label1', 'Split2', 'Index2', 'Label2'])
        
        if all_duplicates:
            print(f"\n✗ Found {len(all_duplicates)} unique duplicate hash(es) across all data:")
            for hash_val, samples in all_duplicates:
                # Write all combinations
                for i in range(len(samples)):
                    for j in range(i+1, len(samples)):
                        s1, s2 = samples[i], samples[j]
                        writer.writerow([
                            hash_val[:12] + '...',  # Truncate hash for readability
                            s1['split'], s1['index'], s1['label'],
                            s2['split'], s2['index'], s2['label']
                        ])
                
                # Print summary
                splits_involved = set(s['split'] for s in samples)
                print(f"  - Hash {hash_val[:12]}... appears {len(samples)} times "
                      f"across splits: {', '.join(sorted(splits_involved))}")
        else:
            print("\n✓ No duplicates found - all samples are unique!")
            writer.writerow(['No leakage detected', '', '', '', '', '', ''])
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("LEAKAGE DETECTION SUMMARY")
    print("="*80)
    
    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    
    if total_overlap == 0 and not all_duplicates:
        print("✓✓✓ NO DATA LEAKAGE DETECTED ✓✓✓")
        print("All train/val/test splits are properly disjoint.")
        print("="*80)
        return 0
    else:
        print("✗✗✗ DATA LEAKAGE DETECTED ✗✗✗")
        print(f"Cross-split overlaps found: {total_overlap}")
        print(f"Total unique duplicate hashes: {len(all_duplicates)}")
        print("="*80)
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

