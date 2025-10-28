import pandas as pd
from dataset_siam import make_balanced_pairs
import pickle
import cv2
import numpy as np
import os

#GG
# Simple image loader
def _open_image(img_name):
    path = os.path.join("/content/drive/MyDrive/siam2/images", f"{img_name}.jpg")
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (105, 105))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

# Load your CSV with labels
df = pd.read_csv("/content/drive/MyDrive/siam2/Groundtruth.csv")

# Generate 5000 balanced pairs
x1, x2, y = make_balanced_pairs(
    df['image_name'].values,
    df['target'].values,
    _open_image,
    target_pairs=5000
)

# Save them to a pickle for training
with open("siamese_isic_split.pkl", "wb") as f:
    pickle.dump(((x1, x2), y), f)

print(f"✅ Saved siamese_isic_split.pkl with {len(y)} pairs")
