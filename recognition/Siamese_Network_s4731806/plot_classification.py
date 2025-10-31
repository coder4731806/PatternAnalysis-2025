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

# Path to the precomputed Siamese pairs pickle
pkl_path = "siamese_isic_split.pkl"  # adjust if needed

# Load the data
with open(pkl_path, "rb") as f:
    data = pickle.load(f)
(x1, x2), y = data

# Simple function to display some random pairs
def show_pairs(x1, x2, y, n=6):
    # Pick n random indices to show
    idxs = random.sample(range(len(y)), n)

    # Prepare subplots: n rows, 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))

    for i, idx in enumerate(idxs):
        # Convert normalized arrays back to images
        img1 = Image.fromarray((x1[idx] * 255).astype(np.uint8))
        img2 = Image.fromarray((x2[idx] * 255).astype(np.uint8))

        # Label based on whether it's a positive or negative pair
        label = "Positive (Same Class)" if y[idx] == 1 else "Negative (Different Class)"

        # Show images side by side
        axes[i, 0].imshow(img1)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Image 1")

        axes[i, 1].imshow(img2)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Image 2\n{label}")

    plt.tight_layout()
    plt.savefig("sample_pairs.png")  # save for quick reference
    print("✅ Saved: sample_pairs.png")

# Display 5 random pairs as a quick check
show_pairs(x1, x2, y, n=5)
