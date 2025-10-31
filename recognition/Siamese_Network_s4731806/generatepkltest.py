import pandas as pd
from dataset_siamese import make_balanced_pairs
import pickle
import cv2
import numpy as np
import os


# Simple helper to load an image by name
def _open_image(img_name):
    path = os.path.join("/content/drive/MyDrive/siam2/images", f"{img_name}.jpg")

    # Make sure the file actually exists
    if not os.path.exists(path):
        return None

    img = cv2.imread(path)

    # Sometimes OpenCV fails to read an image
    if img is None:
        return None

    # Resize to a fixed 105x105 (typical for Siamese networks)
    img = cv2.resize(img, (105, 105))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixels to [0,1]
    return img.astype(np.float32) / 255.0


# Load CSV with image names and labels
df = pd.read_csv("/content/drive/MyDrive/siam2/Groundtruth.csv")

# Generate ~1000 balanced test pairs
x1, x2, y = make_balanced_pairs(
    df['image_name'].values,
    df['target'].values,
    _open_image,
    target_pairs=1000
)

# Save the test pairs to a pickle for later use
with open("/content/drive/MyDrive/siam2/siamese_test.pkl", "wb") as f:
    pickle.dump(((x1, x2), y), f)

print(f"✅ Saved test pickle with {len(y)} pairs")
