import numpy as np
import matplotlib.pyplot as plt
import os

# 1️⃣ Load PathMNIST dataset
data_path = "/projects/weilab/qiongwang/MLO_MAE/PathMNIST/data/pathmnist.npz"
dataset = np.load(data_path)

# Extract splits
splits = {
    "train": (dataset["train_images"], dataset["train_labels"].flatten()),
    "val": (dataset["val_images"], dataset["val_labels"].flatten()),
    "test": (dataset["test_images"], dataset["test_labels"].flatten()),
}

print(f"✅ Loaded PathMNIST: train={splits['train'][0].shape}, val={splits['val'][0].shape}, test={splits['test'][0].shape}")

# 2️⃣ Define function to plot and save 36 samples for one split
def plot_samples(images, labels, split_name):
    unique_classes = np.unique(labels)
    indices = [np.where(labels == c)[0][0] for c in unique_classes]  # one per class

    # Fill up to 36 total
    remaining_needed = 36 - len(indices)
    if remaining_needed > 0:
        extra_indices = np.random.choice(np.arange(len(labels)), remaining_needed, replace=False)
        indices.extend(extra_indices)

    # Plot grid (6×6 = 36)
    fig, axes = plt.subplots(6, 6, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        img = images[indices[i]]
        label = labels[indices[i]]
        ax.imshow(img)
        ax.set_title(f"Class {int(label)}", fontsize=14)
        ax.axis("off")

    plt.suptitle(f"PathMNIST {split_name.capitalize()} Samples (36 Images, All Classes Included)", fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    save_dir = os.path.join(os.path.dirname(data_path), "samples")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split_name}_samples_36.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {split_name} sample grid → {save_path}")

# 3️⃣ Generate for all splits
for split_name, (X, y) in splits.items():
    plot_samples(X, y, split_name)

print("🎉 Done! Generated train/val/test sample grids successfully.")
