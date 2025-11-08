import numpy as np
import matplotlib.pyplot as plt
import os

# 1️⃣ Load dataset
data_path = "/projects/weilab/qiongwang/MLO_MAE/PathMNIST/data/pathmnist.npz"
dataset = np.load(data_path)

X_test = dataset['test_images']
y_test = dataset['test_labels'].flatten()

print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

# 2️⃣ Get unique classes and select one sample per class
unique_classes = np.unique(y_test)
indices = [np.where(y_test == c)[0][0] for c in unique_classes]

# 3️⃣ Add extra random samples to reach 36 total
remaining_needed = 36 - len(indices)
if remaining_needed > 0:
    extra_indices = np.random.choice(np.arange(len(y_test)), remaining_needed, replace=False)
    indices.extend(extra_indices)

# 4️⃣ Plot grid (4×4 = 16 images)
fig, axes = plt.subplots(6, 6, figsize=(20, 20))
for i, ax in enumerate(axes.flat):
    img = X_test[indices[i]]
    label = y_test[indices[i]]
    ax.imshow(img)
    ax.set_title(f"Class {int(label)}", fontsize=16)
    ax.axis("off")

plt.suptitle("PathMNIST Test Samples (36 Images, All Classes Included)", fontsize=28)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 5️⃣ Save image
save_dir = os.path.join(os.path.dirname(data_path), "samples")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "test_samples_36.png")

plt.savefig(save_path, dpi=300)
plt.show()
print(f"✅ Saved to: {save_path}")
