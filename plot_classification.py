"""
plot_classification.py
-----------------------
Visualize a few sample positive (same-class) and negative (different-class)
pairs from the Siamese dataset.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

pkl_path = "siamese_isic_split.pkl"  # adjust as needed

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
(x1, x2), y = data

def show_pairs(x1, x2, y, n=6):
    idxs = random.sample(range(len(y)), n)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    for i, idx in enumerate(idxs):
        img1 = Image.fromarray((x1[idx] * 255).astype(np.uint8))
        img2 = Image.fromarray((x2[idx] * 255).astype(np.uint8))
        label = "Positive (Same Class)" if y[idx] == 1 else "Negative (Different Class)"
        axes[i, 0].imshow(img1); axes[i, 0].axis("off"); axes[i, 0].set_title("Image 1")
        axes[i, 1].imshow(img2); axes[i, 1].axis("off"); axes[i, 1].set_title(f"Image 2\n{label}")
    plt.tight_layout()
    plt.savefig("sample_pairs.png")
    print("✅ Saved: sample_pairs.png")

show_pairs(x1, x2, y, n=5)
