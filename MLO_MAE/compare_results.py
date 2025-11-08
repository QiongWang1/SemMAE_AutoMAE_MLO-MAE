#!/usr/bin/env python3
"""
Compare baseline vs optimized fine-tuning results
"""

import json
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(metrics_path):
    """Load metrics from JSON file"""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Metrics file not found: {metrics_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {metrics_path}")
        return None

def compare_metrics(baseline_metrics, optimized_metrics, output_dir="./DermaMNIST/Output"):
    """Compare and visualize baseline vs optimized results"""
    
    print("\n" + "="*70)
    print("BASELINE VS OPTIMIZED COMPARISON")
    print("="*70)
    
    # Overall accuracy
    baseline_acc = baseline_metrics['test_accuracy']
    optimized_acc = optimized_metrics['test_accuracy']
    improvement = optimized_acc - baseline_acc
    
    print(f"\n📊 Overall Accuracy:")
    print(f"  Baseline:   {baseline_acc:.2f}%")
    print(f"  Optimized:  {optimized_acc:.2f}%")
    print(f"  Change:     {improvement:+.2f}%")
    
    if optimized_acc > baseline_acc:
        print(f"  ✅ IMPROVED by {improvement:.2f}%")
    elif optimized_acc == baseline_acc:
        print(f"  ⚠️  No change")
    else:
        print(f"  ❌ DEGRADED by {abs(improvement):.2f}%")
    
    # F1 Score
    baseline_f1 = baseline_metrics.get('weighted_f1', 0)
    optimized_f1 = optimized_metrics.get('weighted_f1', 0)
    f1_improvement = optimized_f1 - baseline_f1
    
    print(f"\n📈 Weighted F1 Score:")
    print(f"  Baseline:   {baseline_f1:.2f}%")
    print(f"  Optimized:  {optimized_f1:.2f}%")
    print(f"  Change:     {f1_improvement:+.2f}%")
    
    # Per-class comparison
    print(f"\n🎯 Per-Class Accuracy Comparison:")
    print(f"{'Class':<45} {'Baseline':>10} {'Optimized':>10} {'Change':>10}")
    print("-" * 77)
    
    baseline_per_class = baseline_metrics.get('per_class_accuracy', {})
    optimized_per_class = optimized_metrics.get('per_class_accuracy', {})
    
    per_class_changes = {}
    for class_name in baseline_per_class:
        baseline_val = baseline_per_class[class_name]
        optimized_val = optimized_per_class.get(class_name, 0)
        change = optimized_val - baseline_val
        per_class_changes[class_name] = change
        
        # Format class name
        display_name = class_name[:43] if len(class_name) > 43 else class_name
        
        # Emoji for change direction
        emoji = "✅" if change > 1 else ("⚠️" if abs(change) < 0.5 else "❌")
        
        print(f"{display_name:<45} {baseline_val:>9.2f}% {optimized_val:>9.2f}% {change:>+9.2f}% {emoji}")
    
    # Summary statistics
    avg_change = np.mean(list(per_class_changes.values()))
    max_improvement = max(per_class_changes.values())
    max_degradation = min(per_class_changes.values())
    
    print(f"\n📊 Per-Class Statistics:")
    print(f"  Average change:    {avg_change:+.2f}%")
    print(f"  Max improvement:   {max_improvement:+.2f}% ({[k for k,v in per_class_changes.items() if v == max_improvement][0]})")
    print(f"  Max degradation:   {max_degradation:+.2f}% ({[k for k,v in per_class_changes.items() if v == max_degradation][0]})")
    
    # Target achievement
    print(f"\n🎯 Target Achievement:")
    target = 76.8
    if optimized_acc > target:
        margin = optimized_acc - target
        print(f"  ✅ TARGET ACHIEVED! ({optimized_acc:.2f}% > {target:.2f}%)")
        print(f"  Margin: +{margin:.2f}%")
    else:
        gap = target - optimized_acc
        print(f"  ❌ Target not reached ({optimized_acc:.2f}% < {target:.2f}%)")
        print(f"  Gap: -{gap:.2f}%")
    
    # Create visualization
    print(f"\n📊 Generating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall accuracy comparison
    ax1 = axes[0, 0]
    categories = ['Baseline\n76.71%', f'Optimized\n{optimized_acc:.2f}%', 'Target\n76.8%']
    values = [baseline_acc, optimized_acc, target]
    colors = ['#3498db', '#2ecc71' if optimized_acc > baseline_acc else '#e74c3c', '#95a5a6']
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Overall Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(70, 85)
    ax1.axhline(y=target, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Target')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Per-class accuracy comparison
    ax2 = axes[0, 1]
    class_names_short = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    class_names_full = list(baseline_per_class.keys())
    
    baseline_vals = [baseline_per_class[name] for name in class_names_full]
    optimized_vals = [optimized_per_class[name] for name in class_names_full]
    
    x = np.arange(len(class_names_short))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db', alpha=0.7)
    bars2 = ax2.bar(x + width/2, optimized_vals, width, label='Optimized', color='#2ecc71', alpha=0.7)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names_short, fontsize=10)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Improvement by class
    ax3 = axes[1, 0]
    changes = [per_class_changes[name] for name in class_names_full]
    colors_changes = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
    
    bars3 = ax3.bar(class_names_short, changes, color=colors_changes, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Accuracy Change (%)', fontsize=12)
    ax3.set_title('Per-Class Improvement/Degradation', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        label_y = height + 0.3 if height > 0 else height - 0.8
        ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=9)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Baseline', 'Optimized', 'Change'],
        ['Test Accuracy', f'{baseline_acc:.2f}%', f'{optimized_acc:.2f}%', f'{improvement:+.2f}%'],
        ['Weighted F1', f'{baseline_f1:.2f}%', f'{optimized_f1:.2f}%', f'{f1_improvement:+.2f}%'],
        ['Avg Per-Class', f'{np.mean(baseline_vals):.2f}%', f'{np.mean(optimized_vals):.2f}%', 
         f'{np.mean(optimized_vals) - np.mean(baseline_vals):+.2f}%'],
        ['', '', '', ''],
        ['Target Achievement', '', '', ''],
        ['Target Threshold', '76.8%', '', ''],
        ['Achievement', 'N/A', '✅ YES' if optimized_acc > target else '❌ NO', 
         f'{optimized_acc - target:+.2f}%'],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style target section
    for i in range(4):
        table[(5, i)].set_facecolor('#95a5a6')
        table[(5, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Baseline vs Optimized: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, 'baseline_vs_optimized_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {comparison_path}")
    
    plt.close()
    
    # Save comparison report
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE VS OPTIMIZED COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Test Accuracy:  {baseline_acc:.2f}% → {optimized_acc:.2f}% ({improvement:+.2f}%)\n")
        f.write(f"Weighted F1:    {baseline_f1:.2f}% → {optimized_f1:.2f}% ({f1_improvement:+.2f}%)\n\n")
        
        f.write("PER-CLASS ACCURACY\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<45} {'Baseline':>10} {'Optimized':>10} {'Change':>10}\n")
        f.write("-"*70 + "\n")
        for class_name in baseline_per_class:
            baseline_val = baseline_per_class[class_name]
            optimized_val = optimized_per_class.get(class_name, 0)
            change = optimized_val - baseline_val
            f.write(f"{class_name:<45} {baseline_val:>9.2f}% {optimized_val:>9.2f}% {change:>+9.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"TARGET: {'ACHIEVED ✅' if optimized_acc > target else 'NOT REACHED ❌'}\n")
        f.write("="*70 + "\n")
    
    print(f"  ✓ Saved report to: {report_path}")
    
    print("\n" + "="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs optimized results')
    parser.add_argument('--baseline', type=str, 
                       default='./DermaMNIST/Output/evaluation_20251022_105729/metrics/metrics.json',
                       help='Path to baseline metrics JSON')
    parser.add_argument('--optimized', type=str, required=True,
                       help='Path to optimized metrics JSON')
    parser.add_argument('--output_dir', type=str, default='./DermaMNIST/Output',
                       help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    # Load metrics
    print("Loading metrics...")
    baseline_metrics = load_metrics(args.baseline)
    optimized_metrics = load_metrics(args.optimized)
    
    if baseline_metrics is None or optimized_metrics is None:
        print("Error: Could not load one or both metrics files")
        sys.exit(1)
    
    # Compare
    compare_metrics(baseline_metrics, optimized_metrics, args.output_dir)
    
    print("Comparison complete!")

if __name__ == '__main__':
    main()

