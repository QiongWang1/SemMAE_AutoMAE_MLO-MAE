#!/usr/bin/env python3
"""
Real-time training monitoring script for MLO-MAE
Parses log files and displays training progress with plots
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def parse_log_file(log_path):
    """Parse the training log file and extract metrics."""
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return None
    
    data = {
        'steps': [],
        'pretrain_loss': [],
        'finetune_loss': [],
        'mask_loss': [],
        'acc': [],
        'best_acc': [],
        'timestamps': []
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse step and losses
            # [2025-10-20 16:28:25] [INFO] [Problem "pretrain"] [Global Step 10] [Local Step 10] loss: 0.5615279078483582
            pretrain_match = re.search(r'\[Problem "pretrain"\] \[Global Step (\d+)\].*loss: ([\d.]+)', line)
            if pretrain_match:
                step = int(pretrain_match.group(1))
                loss = float(pretrain_match.group(2))
                if step not in data['steps']:
                    data['steps'].append(step)
                    data['pretrain_loss'].append(loss)
            
            finetune_match = re.search(r'\[Problem "finetune"\] \[Global Step (\d+)\].*loss: ([\d.]+)', line)
            if finetune_match:
                step = int(finetune_match.group(1))
                loss = float(finetune_match.group(2))
                # Find corresponding index
                if step in data['steps']:
                    idx = data['steps'].index(step)
                    if len(data['finetune_loss']) <= idx:
                        data['finetune_loss'].append(loss)
                    else:
                        data['finetune_loss'][idx] = loss
            
            mask_match = re.search(r'\[Problem "mask"\] \[Global Step (\d+)\].*loss: ([\d.]+)', line)
            if mask_match:
                step = int(mask_match.group(1))
                loss = float(mask_match.group(2))
                if step in data['steps']:
                    idx = data['steps'].index(step)
                    if len(data['mask_loss']) <= idx:
                        data['mask_loss'].append(loss)
                    else:
                        data['mask_loss'][idx] = loss
            
            # Parse accuracy: acc: 0.9900000000000001 best_acc: 0.9900000000000001
            acc_match = re.search(r'acc: ([\d.]+) best_acc: ([\d.]+)', line)
            if acc_match:
                acc = float(acc_match.group(1))
                best_acc = float(acc_match.group(2))
                data['acc'].append(acc)
                data['best_acc'].append(best_acc)
    
    return data

def create_training_plots(data, output_path='./data/training_progress.png', job_name='MLO-MAE'):
    """Create visualization plots for training progress."""
    
    if not data or len(data['steps']) == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{job_name} Training Progress - CIFAR-100', fontsize=16, fontweight='bold')
    
    # Plot 1: All three losses
    ax1 = axes[0, 0]
    if data['pretrain_loss']:
        ax1.plot(data['steps'][:len(data['pretrain_loss'])], data['pretrain_loss'], 
                label='Pretrain Loss', linewidth=2, color='blue')
    if data['finetune_loss']:
        ax1.plot(data['steps'][:len(data['finetune_loss'])], data['finetune_loss'], 
                label='Finetune Loss', linewidth=2, color='orange')
    if data['mask_loss']:
        ax1.plot(data['steps'][:len(data['mask_loss'])], data['mask_loss'], 
                label='Mask Loss', linewidth=2, color='green')
    ax1.set_xlabel('Global Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Multi-Level Optimization Losses', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pretrain loss only (zoomed)
    ax2 = axes[0, 1]
    if data['pretrain_loss']:
        ax2.plot(data['steps'][:len(data['pretrain_loss'])], data['pretrain_loss'], 
                linewidth=2, color='blue')
        ax2.set_xlabel('Global Step', fontsize=12)
        ax2.set_ylabel('Pretrain Loss', fontsize=12)
        ax2.set_title('MAE Reconstruction Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    ax3 = axes[1, 0]
    if data['acc']:
        steps_acc = list(range(len(data['acc'])))
        ax3.plot(steps_acc, data['acc'], label='Current Acc', 
                linewidth=2, marker='o', markersize=4, color='red')
        ax3.plot(steps_acc, data['best_acc'], label='Best Acc', 
                linewidth=2, marker='s', markersize=4, color='darkred', linestyle='--')
        ax3.set_xlabel('Validation Step', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])
    
    # Plot 4: Training statistics
    ax4 = axes[1, 1]
    stats_text = f"Training Statistics\n" + "="*40 + "\n\n"
    stats_text += f"Total Steps: {len(data['steps'])}\n"
    if data['pretrain_loss']:
        stats_text += f"Pretrain Loss: {data['pretrain_loss'][0]:.4f} → {data['pretrain_loss'][-1]:.4f}\n"
    if data['finetune_loss']:
        stats_text += f"Finetune Loss: {data['finetune_loss'][0]:.4f} → {data['finetune_loss'][-1]:.4f}\n"
    if data['mask_loss']:
        stats_text += f"Mask Loss: {data['mask_loss'][0]:.4f} → {data['mask_loss'][-1]:.4f}\n"
    if data['best_acc']:
        stats_text += f"\nBest Accuracy: {max(data['best_acc']):.2f}%\n"
    stats_text += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training progress plot saved to: {output_path}")

def monitor_training(log_path, output_path, job_name='MLO-MAE', watch=False, interval=60):
    """Monitor training and create plots."""
    
    print("="*60)
    print(f"Monitoring Training: {job_name}")
    print("="*60)
    print(f"Log file: {log_path}")
    print(f"Output plot: {output_path}")
    print("")
    
    if watch:
        import time
        print(f"Watching mode enabled. Updating every {interval} seconds...")
        print("Press Ctrl+C to stop")
        print("")
        
        try:
            while True:
                data = parse_log_file(log_path)
                if data and len(data['steps']) > 0:
                    create_training_plots(data, output_path, job_name)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated plot. Steps: {len(data['steps'])}, " +
                          f"Best acc: {max(data['best_acc']) if data['best_acc'] else 0:.2f}%")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training data...")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring.")
    else:
        # Single update
        data = parse_log_file(log_path)
        if data and len(data['steps']) > 0:
            create_training_plots(data, output_path, job_name)
            print(f"Steps processed: {len(data['steps'])}")
            if data['best_acc']:
                print(f"Best accuracy: {max(data['best_acc']):.2f}%")
        else:
            print("No training data found yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor MLO-MAE training progress')
    parser.add_argument('--log', type=str, default='./checkpoint/log.txt',
                       help='Path to log file')
    parser.add_argument('--output', type=str, default='./data/training_progress.png',
                       help='Path to save plot')
    parser.add_argument('--job_name', type=str, default='MLO-MAE',
                       help='Job name for plot title')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously update plot')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (for watch mode)')
    
    args = parser.parse_args()
    
    monitor_training(args.log, args.output, args.job_name, args.watch, args.interval)

