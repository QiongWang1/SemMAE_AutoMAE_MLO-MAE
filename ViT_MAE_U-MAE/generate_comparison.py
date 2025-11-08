"""
Generate baseline comparison summary from all model results
"""
import os
import json
import csv
import pandas as pd
from pathlib import Path

def extract_metrics_from_csv(csv_path):
    """Extract metrics from CSV file"""
    metrics = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] in ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']:
                key = row[0].replace(' ', '_').replace('(', '').replace(')', '').lower()
                metrics[key] = row[1]
    return metrics

def extract_per_class_metrics(csv_path):
    """Extract per-class metrics"""
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

def generate_comparison_markdown():
    """Generate comparison markdown file"""
    
    base_dir = Path('/projects/weilab/qiongwang/baseline_method/ViT_MAE_U-MAE')
    
    models = {
        'ViT': base_dir / 'ViT' / 'results' / 'metrics_summary.csv',
        'MAE': base_dir / 'MAE' / 'results' / 'metrics_summary.csv',
        'U-MAE': base_dir / 'U-MAE' / 'results' / 'metrics_summary.csv',
    }
    
    # Collect overall metrics
    model_metrics = {}
    per_class_metrics = {}
    
    for model_name, csv_path in models.items():
        if csv_path.exists():
            overall = extract_metrics_from_csv(csv_path)
            model_metrics[model_name] = overall
            
            per_class = extract_per_class_metrics(csv_path)
            per_class_metrics[model_name] = per_class
        else:
            print(f"Warning: {csv_path} not found")
    
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
    
    for model_name, metrics in sorted(model_metrics.items()):
        accuracy = metrics.get('accuracy', 'N/A')
        precision = metrics.get('precision_macro', 'N/A')
        recall = metrics.get('recall_macro', 'N/A')
        f1 = metrics.get('f1_macro', 'N/A')
        
        if model_name == 'ViT':
            notes = 'No pretraining (baseline)'
        elif model_name == 'MAE':
            notes = 'Random masking'
        else:  # U-MAE
            notes = 'Random masking (unified)'
        
        md_content += f"| {model_name} | {accuracy} | {precision} | {recall} | {f1} | {notes} |\n"
    
    md_content += "\n\n## Per-Class Metrics\n\n"
    
    # Per-class comparison
    for model_name, per_class in sorted(per_class_metrics.items()):
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
- **Loss**: Cross-Entropy (for classification)
- **Batch Size**: 128
- **Random Seed**: 42
- **Device**: GPU (A10/A100)

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
    
    print(f"Comparison summary saved to: {output_path}")
    
    # Also generate a CSV summary
    df_data = []
    for model_name, metrics in sorted(model_metrics.items()):
        df_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 'N/A'),
            'Precision_Macro': metrics.get('precision_macro', 'N/A'),
            'Recall_Macro': metrics.get('recall_macro', 'N/A'),
            'F1_Macro': metrics.get('f1_macro', 'N/A'),
        })
    
    df = pd.DataFrame(df_data)
    csv_output = base_dir / 'Baseline_Comparison.csv'
    df.to_csv(csv_output, index=False)
    print(f"CSV summary saved to: {csv_output}")

if __name__ == '__main__':
    generate_comparison_markdown()

