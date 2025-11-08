# Prompt Engineering Playbook for Evaluation of Baseline Methods

## 1. Objective

This playbook provides a standardized workflow and prompt design guide for using Claude Code (or other LLM-based assistants) to reproduce and extend baseline experiments reported in the MLO-MAE paper.

It defines a modular prompting framework for dataset preparation, model configuration, training, evaluation, and result reporting.

The goal is to ensure that future users can efficiently replicate baseline results, adapt the workflow to new datasets, and maintain methodological consistency.

## 2. General Workflow Overview

- **Dataset Preparation**: Ensure the dataset follows the same preprocessing steps (format, normalization, and splits) used in the main experiments.

- **Model Setup**: Configure each baseline architecture (e.g., ViT, MAE, U-MAE, SemMAE, AutoMAE, MLO-MAE) following the standard training hyperparameters described in the paper.

- **Training Execution**: Run the training script using reproducible configuration prompts.

- **Evaluation**: Evaluate each model on the target dataset and compute standard metrics: Accuracy, Precision, Recall, and F1-Score.

- **Result Reporting**: Aggregate results into a comparative performance table and store all logs for traceability.

## 3. Prompt Design Principles

### 3.1 Clarity

- Prompts must be explicit, modular, and environment-agnostic.
- Avoid including local machine details (paths, partitions, GPUs, or dataset IDs).
- Use general variable placeholders for models, datasets, and parameters.

### 3.2 Modularity

- Design prompts so that each stage data preparation, model training, evaluation, and summary can be executed independently or as part of a complete workflow.

### 3.3 Reproducibility

- Always specify random seeds, configuration files, and checkpoint saving rules within prompts.
- Ensure that LLM-generated code outputs can be reproduced without manual editing.

### 3.4 Validation

Include validation prompts for:

- Confirming data integrity (correct file structure and labels).
- Verifying successful model initialization and GPU recognition.
- Confirming the availability of checkpoints before evaluation.

## 4. Prompt Templates

### 4.1 Dataset Preparation Prompt

```
You are an AI coding assistant.  
Prepare the dataset for training baseline models from the MLO-MAE paper.  
Steps:
1. Verify that images and labels are correctly paired and formatted.  
2. Split the dataset into training, validation, and test sets.  
3. Normalize and resize all images according to model input requirements.  
4. Save the processed data in a standardized format (e.g., .pt, .npy, or .csv).  
5. Generate summary statistics (class distribution, image size, normalization parameters).
```

### 4.2 Model Configuration Prompt

```
You are an AI coding assistant.  
Configure a baseline model from the MLO-MAE paper.  
Tasks:
1. Select the architecture: {ViT | MAE | U-MAE | SemMAE | AutoMAE | MLO-MAE}.  
2. Define input size, patch size, masking ratio, and model depth consistent with the paper.  
3. Initialize the model with pretrained weights if applicable.  
4. Set key hyperparameters: optimizer, learning rate, batch size, and total epochs.  
5. Save the configuration as a reusable YAML or JSON file.
```

### 4.3 Training Prompt

```
You are an AI coding assistant.  
Train the selected model using the prepared dataset.  
Steps:
1. Load the model configuration file.  
2. Implement checkpoint saving based on best validation performance.  
3. Enable mixed-precision or gradient accumulation if available.  
4. Monitor loss and metrics at each epoch.  
5. Save logs and checkpoints in an organized experiment directory.
```

### 4.4 Evaluation Prompt

```
You are an AI coding assistant.  
Evaluate a trained baseline model on the test set.  
Tasks:
1. Load the trained checkpoint.  
2. Compute Accuracy, Precision, Recall, and F1-Score.  
3. Optionally compute confusion matrices and per-class metrics.  
4. Export the evaluation results to a structured .csv or .md summary file.
```

### 4.5 Comparative Analysis Prompt

```
You are an AI coding assistant.  
Aggregate and compare results across all baseline models.  
Steps:
1. Load all performance reports generated from evaluation.  
2. Create a Markdown table summarizing Accuracy, Precision, Recall, and F1-Score.  
3. Highlight the best performance per metric.  
4. Optionally generate plots (bar or radar charts) for visualization.
```

### 4.6 Reproducibility Verification Prompt

```
You are an AI coding assistant.  
Verify reproducibility of previous runs.  
Steps:
1. Check that the configuration, code version, and seed match between runs.  
2. Ensure that results fall within the expected performance variance.  
3. Document any deviations or inconsistencies.  
4. Generate a reproducibility summary report in Markdown format.
```

## 5. Output Specification

Each stage must produce the following standardized outputs:

| Stage | Output File | Description |
|-------|-------------|-------------|
| Dataset Preparation | `data_summary.md` | Dataset preprocessing summary |
| Model Configuration | `config.yaml` | Model hyperparameters |
| Training | `training_log.txt` | Epoch-wise training log |
| Evaluation | `evaluation_results.csv` | Final metrics per model |
| Comparative Analysis | `performance_table.md` | Consolidated results |
| Reproducibility | `reproducibility_report.md` | Consistency verification summary |

## 6. Best Practices

- Maintain consistent naming conventions for models and logs.
- Use version controlled configuration files.
- Keep prompt outputs traceable in separate folders for each run.
- Validate results after every major stage before moving forward.


