# Prompt Engineering Playbook: MLO-MAE for Medical Imaging

**Project**: Multi-Level Optimized Masked Autoencoder (MLO-MAE)  
**Collaboration**: Qiong × Claude Code  
**Result**: 78.20%  Test Accuracy  
**Goal**: Prompt design for AI-assisted reproduction and domain adaptation of a deep-learning framework(MLO_MAE) on medical images.

---

## 0. System Setup Prompt — Role & Objective

```
You are an AI research assistant running on SCC with GPU access (H200 / V100).
Your mission is to reproduce and adapt the MLO-MAE framework from
"Downstream Task Guided Masking Learning in Masked Autoencoders Using Multi-Level Optimization" (TMLR 2024)
to a medical-image classification task using the MedMNIST dataset.


Requirements

- Python 3.10+, PyTorch 2.x, CUDA 12+.
- Reproducible, modular codebase (PEP 8, fixed seeds).
- Three-stage workflow: Pretraining → Finetuning → Evaluation.
- Output complete metrics, confusion matrix, and reconstruction visualization.
- No data leakage between train/val/test.
- HPC compatible (Slurm jobs, multi-GPU).
```
---

## 1. Project Specification Prompt — Scope & Acceptance Criteria

```
Design a reproducible pipeline that trains and evaluates MLO-MAE.

Data & Preprocessing

- Dataset = DermaMNIST (HAM10000 subset, 7 disease classes).
- Input size = 3×32×32 (bicubic resize).
- Normalization: ImageNet mean / std.
- Augmentations: RandomCrop (32, padding=4), RandomHorizontalFlip, Cutout(16).
- Splits: 70% train / 10% val / 20% test with fixed seed.

Modeling

- Backbone: ViT-Base (12 layers, 768 dim, patch size 2×2).
- Decoder: 8-layer lightweight Transformer (512 dim).
- Mask Network: learnable mask probability predictor (256 patches).
- Loss functions: MAE reconstruction + classification cross-entropy.
- Optimizer: AdamW (β₁ = 0.9, β₂ = 0.95).
- Scheduler: Cosine Annealing.
- Mask ratio = 0.75.
- Unroll steps = {pretrain: 1, finetune: 1, mask: 1}.

Acceptance Criteria

- Test accuracy ≥ 75%.
- Weighted F1 ≥ 74%.
- Successful generation of:
  - `confusion_matrix.png`
  - `per_class_accuracy.png`
  - `reconstruction_samples.png`
  - `metrics.json`
  - `LATEST_REPORT.md`
```
---

## 2. Data Preparation Prompt

```
Generate `data/load_dermamnist.py` that:

1. Loads DermaMNIST (train/val/test) via `medmnist`.
2. Applies resize → normalization → augmentation transforms.
3. Builds PyTorch DataLoaders (`batch_size = 64`).
4. Logs dataset statistics (sample count, class distribution).
5. Saves split info to `data/stats_derma.json`.

Add unit test `tests/test_dataloader.py` verifying:
- Shape = `[batch, 3, 32, 32]`.
- Label range = 0 – 6.
- Deterministic lengths across runs.
```
---

## 3. Model Architecture Prompt

```
Implement in `models/mlomae_derma.py`:

Classes

- `MaskingNetwork`: MLP (input = 256×768, hidden = 512, output = 256 sigmoid).
- `MAEEncoder`: ViT-Base 12×12×768.
- `MAEDecoder`: Lightweight 8×512.
- `ClassifierHead`: Linear(768→7) + Dropout 0.1.

Forward Flow

input → masking network (T) → masked patches → encoder (E) → decoder (D)
                                     ↓
                                 classifier (C)


Interfaces

def forward(self, images, labels=None, mask_ratio=0.75): ...
def training_step(self, batch, stage): ...


All modules registered in `models/__init__.py` via `build_model(cfg)`.
```


---

## 4. Training Pipeline Prompt

```
Implement `scripts/train_derma.py` with arguments:

```bash
--stage {pretrain, finetune}
--epochs 200
--batch_size 32
--lr 1e-3
--mask_ratio 0.75
--unroll_steps_pretrain 1
--unroll_steps_mask 1
--output_dir ./Output/pretrain_YYYYMMDD
```
```
Procedures

1. Pretrain encoder (E) + mask network (T) on masked images.
2. Save checkpoint `pretrain_best.pth`.
3. Finetune classifier (C) end-to-end on full images.
4. Log loss & metrics to WandB (project = "MLO-MAE-DermaMNIST").
5. After finetuning, call evaluation script automatically.

Output Structure


Output/
├── pretrain_*/checkpoints/
├── finetune_*/checkpoints/mlomae-ckpt.t7
├── evaluation_*/metrics.json
├── visualizations/*.png
└── LATEST_REPORT.md

```

---

## 5. Evaluation & Visualization Prompt

```
Implement `scripts/evaluate_derma.py` that:

1. Loads checkpoint `mlomae-ckpt.t7`.
2. Computes Accuracy, Weighted F1, Precision, Recall.
3. Generates `confusion_matrix.png`, `per_class_accuracy.png`.
4. Saves results to `evaluation_*/metrics.json`.
5. Creates Markdown summary (`LATEST_REPORT.md`) with tables and plots.

Include function `visualize_reconstructions()` to output 8 samples (Original | Masked | Reconstructed).
```
---

## 6. HPC / Slurm Integration Prompt
```
Create `jobs/train_mlomae.slurm`:

```bash
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=mlomae_derma
#SBATCH --output=logs/%x_%j.out


Workflow

1. `conda activate /projects/weilab/qiongwang/envs/mae`
2. `python scripts/train_derma.py --stage pretrain ...`
3. `python scripts/train_derma.py --stage finetune ...`
4. After finetuning, auto-run `evaluate_derma.py`.
5. Upload logs and metrics to WandB.
```
---

## 7. Performance & Acceptance Validation Prompt
```
Validate that the pipeline produces:

- Accuracy = 
- Weighted F1 = 
- Reconstruction images visually preserve lesion features.
- Training time ≈ 

Store summary in `docs/RESULTS.md`.
```
---

## 8. Ablation & Extension Prompt
```
Add configurations under `configs/ablation/`:

- Mask ratio = 0.5, 0.9.
- Patch size = 4×4.
- Semantic Guided Masking (biased mask based on lesion regions).

Compare accuracy changes and plot in `docs/ABLATION.md`.

Future datasets for reuse:
- PathMNIST (histo-patch classification)
- OrganMNIST (T1 MRI organ identification)
```
---

## 9. Quality & Reproducibility Checklist Prompt
```
Create `checklist.md` confirming:

- [x] Fixed random seed = 42.
- [x] Deterministic DataLoader behavior.
- [x] Independent val/test splits.
- [x] Metrics logged in JSON + WandB.
- [x] flake8 compliant code.
- [x] Unit tests for data and model.
- [x] All plots and reports auto-generated.
```
---

## 10. Reporting & Model Card Prompt
```
Generate:

1. `MODEL_CARD.md` — task, architecture, dataset license, metrics, limitations.
2. `REPORT.md` — summary of training process and visual results.
3. `docs/INTERPRETABILITY.md` — explain mask focus and semantic regions.

Format in scientific style for reproducibility and transparency.
```
---

## 11. Meta-Prompt for Future Experiments
```
When extending to other medical datasets (e.g., BloodMNIST or OrganMNIST):


You are an AI research assistant.
Your task is to adapt MLO-MAE to the dataset <DATASET_NAME>,
maintaining the same framework structure as DermaMNIST.
Only update num_classes, input_size, and dataset loader.
All other settings remain unchanged unless justified.


Output must include:
- Full training log.
- `metrics.json`.
- visualizations.
- summary report.
```
---

## 12. Prompt for Improve Experiments Performance
```
Goal:
Improve MLO-MAE performance on DermaMNIST to surpass 76.8% test accuracy with minimal cost (no architecture change, no large-scale retraining).

Context:
Current result = 76.71% (ViT-Base encoder, mask ratio=0.75, image size=32×32, pretrain 6.5h + finetune 1.5h on 2×H200 GPUs).

Task:
1. Focus on lightweight improvements to enhance accuracy and stability.
2. Do NOT change model architecture or dataset.
3. Modify only training configuration and optimization strategy.

Optimization option:
- Reduce mask_ratio from 0.75 → 0.65 to preserve lesion details.
- Increase input image size to 64×64 for finer texture capture.
- Extend fine-tuning epochs (e.g., 30 → 60).
- Use cosine LR scheduler with warmup (5 epochs).
- Add CutMix or Mixup augmentation.
- Enable class reweighting or focal loss to handle imbalance.

Deliverables:
- Update the training configuration (YAML or script).
- Re-run fine-tuning using the pre-trained checkpoint (avoid full re-pretrain).
- Report final test accuracy, F1-score, and per-class results.
- Generate brief analysis on which change most contributed

Command examples to generate:
- bash train_derma_tuned.sh (with logging)
- tensorboard visualization link
- auto-checkpoint selection (best.ckpt or last.ckpt)

Output format:
A single unified plan with modified parameters, estimated runtime, and expected accuracy improvement.
```