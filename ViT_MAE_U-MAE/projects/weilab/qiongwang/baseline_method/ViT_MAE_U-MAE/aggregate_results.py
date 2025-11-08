#!/usr/bin/env python
"""
Aggregate results from all three baseline models
"""
import os
import csv
import pandas as pd
from pathlib import Path

def extract_metrics_from_csv(csv_path):
    """Extract metrics from CSV file"""
    if not os.path.exists(csv_path):
        return None
    
    metrics = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] in ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']:
                metrics[row[0]] = row[1]
    return metrics

def extract_per_class_metrics(csv_path):
    """Extract per-class metrics"""
    if not os.path.exists(csv_path):
        return None
    
    per_class = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        
        # Find the per-class section
        header_idx = None
        for i, line in enumerate(lines):
            if len(line) > 0 and line[0] == 'Class':
                header_idx = i
                break
        
        if header_idx is not None:
            for i in range(header_idx + 1, len(lines)):
                if len(lines[i]) >= 5:
                    per_class.append({
                        'class': lines[i][0],
                        'precision': lines[i][1],
                        'recall': lines[i][2],
                        'f1': lines[i][3],
                        'support': lines[i][4]
                    })
    
    return per_class

def main():
    """Aggregate results from all models"""
    base_dir = Path(__file__).parent
    
    models = {
        'ViT': base_dir / 'ViT' / 'results' / 'metrics_summary.csv',
        'MAE': base_dir / 'MAE' / 'results' / 'metrics_summary.csv',
        'U-MAE': base_dir / 'U-MAE' / 'results' / 'metrics_summary.csv',
    }
    
    print("=" * 60)
    print("Aggregating Results from Baseline Models")
    print("=" * 60)
    print()
    
    # Collect overall metrics
    model_data = {}
    per_class_data = {}
    
    for model_name, csv_path in models.items():
        print(f"Checking {model_name}...")
        if csv_path.exists():
            overall = extract_metrics_from_csv(csv_path)
            if overall:
                model_data[model_name] = overall
                print(f"  ✓ Found metrics")
            else:
                print(f"  ✗ No metrics found")
            
            per_class = extract_per_class_metrics(csv_path)
            if per_class:
                per_class_data[model_name] = per_class
        else:
            print(f"  ✗ {csv_path} not found")
    
    if not model_data:
        print("\nERROR: No results found for any model!")
        print("Please ensure at least one model has completed training.")
        return 1
    
    print()
    
    # Generate markdown
    md_content = """# Baseline Model Comparison on DermaMNIST

This report compares the performance of three baseline models for skin lesion classification on the DermaMNIST dataset.

## Dataset Information

- **Dataset**: DermaMNIST (from MedMNIST)
- **Input Size**: 32×32 (resized from 28×28)
- **Classes**: 7 skin lesion types
- **Splits**: Train=7007, Val=1003, Test=2005
- **Task**: Skin lesion classification

## Overall Metrics

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | Notes |
|-------|----------|-------------------|----------------|------------|-------|
"""
    
    for model_name in sorted(model_data.keys()):
        metrics = model_data[model_name]
        accuracy = metrics.get('Accuracy', 'N/A')
        precision = metrics.get('Precision (Macro)', 'N/A')
        recall = metrics.get('Recall (Macro)', 'N/A')
        f1 = metrics.get('F1 (Macro)', 'N/A')
        
        if model_name == 'ViT':
            notes = 'No pretraining (baseline)'
        elif model_name == 'MAE':
            notes = 'Random masking'
        else:  # U-MAE
            notes = 'Random masking (unified)'
        
        md_content += f"| {model_name} | {accuracy} | {precision} | {recall} | {f1} | {notes} |\n"
    
    md_content += "\n\n## Per-Class Metrics\n\n"
    
    # Per-class comparison
    for model_name in sorted(per_class_data.keys()):
        if model_name not in per_class_data:
            continue
            
        per_class = per_class_data[model_name]
        md_content += f"### {model_name}\n\n"
        md_content += "| Class | Precision | Recall | F1 | Support |\n"
        md_content += "|-------|-----------|--------|------|---------|\n"
        
        for cls in per_class:
            md_content += f"| {cls['class']} | {cls['precision']} | {cls['recall']} | {cls['f1']} | {cls['support']} |\n"
        
        md_content += "\n"
    
    md_content += """## Training Details

All models were trained with:
- **Optimizer**: Adam / AdamW
- **LR Schedule**: Cosine Annealing
- **Loss**: Cross-Entropy (for classification), MSE (for pretraining)
- **Batch Size**: 128
- **Random Seed**: 42
- **Device**: GPU (A10/A100 on SCC)

### Model Architectures

- **ViT**: Standard Vision Transformer (6 blocks, 8 heads, embed_dim=192)
- **MAE**: Masked Autoencoder with random masking (75% mask ratio)
- **U-MAE**: Unified Masked Autoencoder with similar masking strategy

### Training Phases

- **ViT**: Direct classification training (100 epochs)
- **MAE & U-MAE**: Two-phase training
  - Phase 1: Self-supervised pretraining (50 epochs)
  - Phase 2: Supervised finetuning (100 epochs)

"""
    
    # Save markdown
    output_path = base_dir / 'Baseline_Comparison.md'
    with open(output_path, 'w') as f:
        f.write(md_content)
    
    print(f"✓ Comparison summary saved to: {output_path}")
    
    # Also generate a CSV summary
    df_data = []
    for model_name in sorted(model_data.keys()):
        metrics = model_data[model_name]
        df_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('Accuracy', 'N/A'),
            'Precision_Macro': metrics.get('Precision (Macro)', 'N/A'),
            'Recall_Macro': metrics.get('Recall (Macro)', 'N/A'),
            'F1_Macro': metrics.get('F1 (Macro)', 'N/A'),
        })
    
    df = pd.DataFrame(df_data)
    csv_output = base_dir / 'Baseline_Comparison.csv'
    df.to_csv(csv_output, index=False)
    print(f"✓ CSV summary saved to: {csv_output}")
    
    print()
    print("=" * 60)
    print("Aggregation complete!")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    exit(main())

