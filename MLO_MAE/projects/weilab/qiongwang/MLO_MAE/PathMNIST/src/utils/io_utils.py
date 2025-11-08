"""
I/O utilities for PathMNIST MLO-MAE pipeline
Handles output directory structure, metrics saving, and report generation
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def make_output_tree(job_id, base_dir='/projects/weilab/qiongwang/MLO_MAE/PathMNIST/output'):
    """
    Create standardized output directory tree for a SLURM job
    
    Structure:
    output/${SLURM_JOB_ID}/
      ├─ job_log/              # all raw logs
      ├─ checkpoint/           # intermediate checkpoints
      ├─ best_checkpoint/      # best model ckpt + metrics.json
      ├─ reports/              # _Results.md
      └─ preds/                # predictions/softmax
    
    Args:
        job_id: SLURM job ID
        base_dir: base output directory
    
    Returns:
        dict with paths to all output subdirectories
    """
    output_root = os.path.join(base_dir, str(job_id))
    
    subdirs = {
        'root': output_root,
        'job_log': os.path.join(output_root, 'job_log'),
        'checkpoint': os.path.join(output_root, 'checkpoint'),
        'best_checkpoint': os.path.join(output_root, 'best_checkpoint'),
        'reports': os.path.join(output_root, 'reports'),
        'preds': os.path.join(output_root, 'preds'),
    }
    
    # Create all directories
    for name, path in subdirs.items():
        os.makedirs(path, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"Output Directory Tree Created for Job {job_id}")
    print("=" * 70)
    for name, path in subdirs.items():
        print(f"  {name:20s}: {path}")
    print("=" * 70 + "\n")
    
    return subdirs


def save_metrics(metrics_dict, output_path):
    """
    Save metrics to JSON file
    
    Args:
        metrics_dict: dictionary of metrics
        output_path: path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    clean_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            clean_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            clean_metrics[key] = value.tolist()
        else:
            clean_metrics[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(clean_metrics, f, indent=4)
    
    print(f"✓ Metrics saved to: {output_path}")


def compute_metrics(y_true, y_pred, split='test'):
    """
    Compute classification metrics
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        split: name of the split (for logging)
    
    Returns:
        dict with accuracy, precision, recall, f1
    """
    # Convert to numpy if needed
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    
    metrics = {
        'split': split,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: dict with metrics
    """
    print("\n" + "=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    print(f"Split:      {metrics.get('split', 'N/A')}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print("=" * 70 + "\n")


def write_markdown_report(metrics, output_path, additional_info=None):
    """
    Write a markdown report with metrics table
    
    Args:
        metrics: dict with metrics
        output_path: path to save markdown file
        additional_info: optional dict with additional information
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# PathMNIST MLO-MAE Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if additional_info:
            f.write("## Configuration\n\n")
            for key, value in additional_info.items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")
        
        f.write("## Test Set Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {metrics['recall']:.4f} |\n")
        f.write(f"| F1 Score | {metrics['f1']:.4f} |\n")
        f.write("\n")
        
        # Additional table format as requested
        f.write("## Results Table\n\n")
        f.write("| Split | Accuracy | Precision | Recall | F1 |\n")
        f.write("|------:|---------:|----------:|-------:|---:|\n")
        f.write(f"| {metrics.get('split', 'Test'):5s} | {metrics['accuracy']:8.4f} | "
                f"{metrics['precision']:9.4f} | {metrics['recall']:6.4f} | "
                f"{metrics['f1']:6.4f} |\n")
    
    print(f"✓ Report saved to: {output_path}")


def copy_log_files(source_patterns, dest_dir):
    """
    Copy log files to job_log directory
    
    Args:
        source_patterns: list of file patterns to copy
        dest_dir: destination directory
    """
    import glob
    
    os.makedirs(dest_dir, exist_ok=True)
    
    for pattern in source_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            if os.path.isfile(file_path):
                dest_path = os.path.join(dest_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                print(f"✓ Copied {file_path} to {dest_path}")


def save_predictions(y_true, y_pred, y_prob, output_dir, split='test'):
    """
    Save predictions and probabilities to files
    
    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted labels (N,)
        y_prob: prediction probabilities (N, C)
        output_dir: directory to save predictions
        split: name of the split
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    if hasattr(y_prob, 'cpu'):
        y_prob = y_prob.cpu().numpy()
    
    # Save predictions
    pred_path = os.path.join(output_dir, f'{split}_predictions.npz')
    np.savez(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )
    
    print(f"✓ Predictions saved to: {pred_path}")
    
    # Also save as text for easy inspection
    txt_path = os.path.join(output_dir, f'{split}_predictions.txt')
    with open(txt_path, 'w') as f:
        f.write("idx,true_label,pred_label,correct\n")
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            correct = '✓' if true == pred else '✗'
            f.write(f"{i},{true},{pred},{correct}\n")
    
    print(f"✓ Predictions text saved to: {txt_path}")


def parse_hyperparameters(log_file, stage='pretrain'):
    """
    Parse hyperparameters from DermaMNIST log files
    
    Args:
        log_file: path to log file
        stage: 'pretrain' or 'finetune'
    
    Returns:
        dict of hyperparameters
    """
    hparams = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        if stage == 'pretrain':
            # Parse Namespace(...) from pretrain log
            if 'Namespace(' in content:
                namespace_start = content.find('Namespace(')
                namespace_end = content.find(')', namespace_start)
                namespace_str = content[namespace_start:namespace_end+1]
                
                # Extract key hyperparameters
                import re
                hparams['batch_size'] = int(re.search(r'batch_size=(\d+)', namespace_str).group(1))
                hparams['epochs'] = int(re.search(r'epochs=(\d+)', namespace_str).group(1))
                hparams['lr'] = float(re.search(r'lr=([\d.e-]+)', namespace_str).group(1))
                hparams['weight_decay'] = float(re.search(r'weight_decay=([\d.e-]+)', namespace_str).group(1))
                hparams['warmup_epochs'] = int(re.search(r'warmup_epochs=(\d+)', namespace_str).group(1))
                hparams['mask_ratio'] = float(re.search(r'mask_ratio=([\d.]+)', namespace_str).group(1))
                hparams['input_size'] = int(re.search(r'input_size=(\d+)', namespace_str).group(1))
                
        elif stage == 'finetune':
            # Parse Namespace(...) from finetune log
            if 'Namespace(' in content:
                namespace_start = content.find('Namespace(')
                namespace_end = content.find(')', namespace_start)
                namespace_str = content[namespace_start:namespace_end+1]
                
                # Extract key hyperparameters
                import re
                hparams['lr'] = float(re.search(r'lr=([\d.e-]+)', namespace_str).group(1))
                hparams['weight_decay'] = float(re.search(r'weight_decay=([\d.e-]+)', namespace_str).group(1))
                hparams['n_epochs'] = int(re.search(r'n_epochs=(\d+)', namespace_str).group(1))
                bs_match = re.search(r"bs='(\d+)'", namespace_str)
                if bs_match:
                    hparams['batch_size'] = int(bs_match.group(1))
                size_match = re.search(r"size='(\d+)'", namespace_str)
                if size_match:
                    hparams['input_size'] = int(size_match.group(1))
    
    except Exception as e:
        print(f"Warning: Could not parse hyperparameters from {log_file}: {e}")
        # Return defaults
        if stage == 'pretrain':
            hparams = {
                'batch_size': 32,
                'epochs': 200,
                'lr': 0.0001,
                'weight_decay': 0.001,
                'warmup_epochs': 5,
                'mask_ratio': 0.75,
                'input_size': 32
            }
        else:
            hparams = {
                'lr': 0.0001,
                'weight_decay': 5e-5,
                'n_epochs': 1000,
                'batch_size': 64,
                'input_size': 32
            }
    
    return hparams


if __name__ == '__main__':
    # Test utilities
    print("Testing I/O Utilities...")
    print("=" * 70)
    
    # Test output tree creation
    job_id = 'test_12345'
    output_dirs = make_output_tree(job_id, base_dir='./test_output')
    
    # Test metrics computation and saving
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    
    metrics = compute_metrics(y_true, y_pred, split='test')
    print_metrics(metrics)
    
    # Save metrics
    metrics_path = os.path.join(output_dirs['best_checkpoint'], 'metrics.json')
    save_metrics(metrics, metrics_path)
    
    # Write report
    report_path = os.path.join(output_dirs['reports'], '_Results.md')
    additional_info = {
        'Job ID': job_id,
        'Model': 'MLO-MAE',
        'Dataset': 'PathMNIST',
        'Epochs': 100
    }
    write_markdown_report(metrics, report_path, additional_info)
    
    # Test prediction saving
    y_prob = np.random.rand(len(y_true), 9)
    save_predictions(y_true, y_pred, y_prob, output_dirs['preds'], split='test')
    
    print("\n✓ I/O utilities test passed!")
    
    # Clean up test output
    import shutil
    shutil.rmtree('./test_output', ignore_errors=True)


