import pandas as pd
from recognition.Siamese_Network_s4731806.dataset import make_balanced_pairs
import pickle
import cv2
import numpy as np
import os


# GG: simple helper to load an image by name
def _open_image(img_name):
    path = os.path.join("/content/drive/MyDrive/siam2/images", f"{img_name}.jpg")

    # Make sure the file exists
    if not os.path.exists(path):
        return None

    img = cv2.imread(path)

    # Some images might fail to load
    if img is None:
        return None

    # Resize to a fixed size (Siamese networks like small consistent images)
    img = cv2.resize(img, (105, 105))

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0,1] floats
    return img.astype(np.float32) / 255.0


# Load the CSV file that contains image names and their labels
df = pd.read_csv("/content/drive/MyDrive/siam2/Groundtruth.csv")

# Make 5000 positive & negative pairs for training
x1, x2, y = make_balanced_pairs(
    df['image_name'].values,
    df['target'].values,
    _open_image,
    target_pairs=5000
)

# Dump the pairs into a pickle so the Siamese network can load them quickly
with open("siamese_isic_split.pkl", "wb") as f:
    pickle.dump(((x1, x2), y), f)

print(f"✅ Saved siamese_isic_split.pkl with {len(y)} pairs")
