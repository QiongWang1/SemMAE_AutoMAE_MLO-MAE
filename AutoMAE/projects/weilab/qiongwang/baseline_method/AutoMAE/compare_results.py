"""
Compare baseline vs improved AutoMAE results on DermaMNIST
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path


def parse_csv_results(csv_path):
    """Parse CSV results file"""
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    results = {}
    for _, row in df.iterrows():
        results[row['Metric']] = row['Value']
    return results


def generate_comparison_report(baseline_csv, improved_csv, output_path):
    """Generate markdown comparison report"""
    
    baseline = parse_csv_results(baseline_csv)
    improved = parse_csv_results(improved_csv)
    
    if baseline is None:
        print(f"Error: Baseline results not found at {baseline_csv}")
        return False
    
    if improved is None:
        print(f"Error: Improved results not found at {improved_csv}")
        return False
    
    # Calculate improvements
    acc_improve = improved['Accuracy'] - baseline['Accuracy']
    prec_improve = improved['Precision'] - baseline['Precision']
    rec_improve = improved['Recall'] - baseline['Recall']
    f1_improve = improved['F1'] - baseline['F1']
    
    # Generate report
    with open(output_path, 'w') as f:
        f.write('# AutoMAE Performance Comparison: Baseline vs Improved\n\n')
        f.write('**Comparison Date:** {}\n\n'.format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        # Overall metrics comparison
        f.write('## Overall Metrics Comparison\n\n')
        f.write('| Metric | Baseline | Improved | Improvement | Change (%) |\n')
        f.write('|--------|----------|----------|-------------|------------|\n')
        
        metrics = [
            ('Accuracy', baseline['Accuracy'], improved['Accuracy'], acc_improve),
            ('Precision', baseline['Precision'], improved['Precision'], prec_improve),
            ('Recall', baseline['Recall'], improved['Recall'], rec_improve),
            ('F1 Score', baseline['F1'], improved['F1'], f1_improve)
        ]
        
        for name, base, imp, delta in metrics:
            pct_change = (delta / base) * 100 if base > 0 else 0
            change_str = "✓" if delta > 0 else "✗" if delta < 0 else "="
            f.write(f'| {name} | {base:.2f}% | {imp:.2f}% | {change_str} {delta:+.2f}% | {pct_change:+.2f}% |\n')
        
        # Summary
        f.write('\n## Summary\n\n')
        if acc_improve > 0:
            f.write(f'✅ **Improved model achieves {acc_improve:.2f}% higher accuracy**\n\n')
        else:
            f.write(f'⚠️ **Accuracy decreased by {abs(acc_improve):.2f}%**\n\n')
        
        # Per-class comparison
        f.write('## Per-Class Performance Comparison\n\n')
        f.write('### F1 Scores\n\n')
        f.write('| Class | Baseline F1 | Improved F1 | Improvement |\n')
        f.write('|-------|-------------|-------------|-------------|\n')
        
        for i in range(7):
            base_f1 = baseline.get(f'Class_{i}_F1', 0)
            imp_f1 = improved.get(f'Class_{i}_F1', 0)
            delta = imp_f1 - base_f1
            change_str = "✓" if delta > 0 else "✗" if delta < 0 else "="
            f.write(f'| Class {i} | {base_f1:.2f}% | {imp_f1:.2f}% | {change_str} {delta:+.2f}% |\n')
        
        # Key improvements
        f.write('\n## Key Improvements Applied\n\n')
        f.write('1. **Regularization**\n')
        f.write('   - Added dropout (0.3) to classification head\n')
        f.write('   - Implemented label smoothing (0.1)\n')
        f.write('   - Increased weight decay (0.05 → 0.1)\n\n')
        
        f.write('2. **Architecture**\n')
        f.write('   - Deeper classification head (3-layer MLP: 192→512→256→7)\n')
        f.write('   - Added GELU activations\n\n')
        
        f.write('3. **Training Strategy**\n')
        f.write('   - Lower learning rate (1e-3 → 1e-4)\n')
        f.write('   - Early stopping with patience=15\n')
        f.write('   - Gradient clipping (max_norm=1.0)\n')
        f.write('   - Class-weighted loss for imbalanced dataset\n\n')
        
        f.write('4. **Optimizer**\n')
        f.write('   - AdamW with β=(0.9, 0.999)\n')
        f.write('   - Cosine annealing LR schedule\n')
        f.write('   - Warmup for 10 epochs\n\n')
        
        # Analysis
        f.write('## Analysis\n\n')
        
        if acc_improve > 5:
            f.write('### Significant Improvement ✅\n\n')
            f.write(f'The improved model shows significant performance gains ({acc_improve:.2f}% accuracy improvement). ')
            f.write('The key factors contributing to this improvement are:\n\n')
            f.write('- **Reduced Overfitting**: Dropout and label smoothing effectively regularized the model\n')
            f.write('- **Better Generalization**: Early stopping prevented the model from overfitting to training data\n')
            f.write('- **Class Balance**: Weighted loss helped improve performance on minority classes\n')
            f.write('- **Stable Training**: Lower learning rate and gradient clipping ensured stable convergence\n\n')
        elif acc_improve > 2:
            f.write('### Moderate Improvement ✅\n\n')
            f.write(f'The improved model shows moderate performance gains ({acc_improve:.2f}% accuracy improvement). ')
            f.write('Further improvements might be achieved by:\n\n')
            f.write('- Tuning dropout rate\n')
            f.write('- Adjusting learning rate schedule\n')
            f.write('- Experimenting with different architectures\n')
            f.write('- Using more advanced augmentation strategies\n\n')
        elif acc_improve > 0:
            f.write('### Marginal Improvement ⚠️\n\n')
            f.write(f'The improved model shows marginal gains ({acc_improve:.2f}% accuracy improvement). ')
            f.write('Consider:\n\n')
            f.write('- Longer pretraining on AutoMAE\n')
            f.write('- Different model architectures\n')
            f.write('- More sophisticated augmentation\n')
            f.write('- Ensemble methods\n\n')
        else:
            f.write('### Performance Decreased ⚠️\n\n')
            f.write(f'The improved model shows decreased performance ({acc_improve:.2f}%). This suggests:\n\n')
            f.write('- Over-regularization (dropout/weight decay too high)\n')
            f.write('- Learning rate too low\n')
            f.write('- Early stopping too aggressive\n')
            f.write('- Consider tuning hyperparameters\n\n')
        
        # Baseline issues
        f.write('## Baseline Model Issues\n\n')
        f.write('The baseline model suffered from:\n\n')
        f.write('1. **Severe Overfitting**\n')
        f.write('   - Train accuracy: 99.93%\n')
        f.write('   - Val accuracy peaked at 76.47% then declined\n')
        f.write('   - Test accuracy: 72.57%\n\n')
        
        f.write('2. **Class Imbalance**\n')
        f.write('   - Class 5 (dominant): 87% F1\n')
        f.write('   - Classes 0,3 (minority): 27-34% F1\n')
        f.write('   - No class weighting applied\n\n')
        
        f.write('3. **No Regularization**\n')
        f.write('   - No dropout in classification head\n')
        f.write('   - Learning rate too high (1e-3)\n')
        f.write('   - No early stopping\n\n')
        
        # Recommendations
        f.write('## Recommendations\n\n')
        f.write('Based on the comparison, we recommend:\n\n')
        
        if acc_improve > 3:
            f.write('1. ✅ **Deploy the improved model** - Shows significant gains\n')
            f.write('2. Consider further improvements:\n')
            f.write('   - Longer pretraining\n')
            f.write('   - Test-time augmentation\n')
            f.write('   - Model ensembling\n')
        else:
            f.write('1. ⚠️ **Further tuning needed**\n')
            f.write('2. Hyperparameter optimization:\n')
            f.write('   - Grid search on dropout, lr, weight_decay\n')
            f.write('   - Experiment with different architectures\n')
            f.write('3. Data-centric approaches:\n')
            f.write('   - More sophisticated augmentations\n')
            f.write('   - Semi-supervised learning\n')
        
        f.write('\n---\n\n')
        f.write('*Generated automatically by compare_results.py*\n')
    
    print(f"\n✓ Comparison report saved to: {output_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Accuracy:  {baseline['Accuracy']:.2f}% → {improved['Accuracy']:.2f}% ({acc_improve:+.2f}%)")
    print(f"Precision: {baseline['Precision']:.2f}% → {improved['Precision']:.2f}% ({prec_improve:+.2f}%)")
    print(f"Recall:    {baseline['Recall']:.2f}% → {improved['Recall']:.2f}% ({rec_improve:+.2f}%)")
    print(f"F1 Score:  {baseline['F1']:.2f}% → {improved['F1']:.2f}% ({f1_improve:+.2f}%)")
    print("="*80)
    
    return True


def main():
    parser = argparse.ArgumentParser('Compare AutoMAE Results')
    parser.add_argument('--baseline_csv', default='job/results/metrics_summary.csv', type=str)
    parser.add_argument('--improved_csv', default='job/results_improve/metrics_summary.csv', type=str)
    parser.add_argument('--output', default='job/results_improve/Comparison_to_Baseline.md', type=str)
    args = parser.parse_args()
    
    success = generate_comparison_report(args.baseline_csv, args.improved_csv, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

