"""
dataset_siamese.py
Unified dataset loader for Siamese Network (PyTorch)
----------------------------------------------------
Loads .pkl files containing precomputed (x1, x2, y) pairs
and prepares them for training or validation.

Author: Harshit Vishwa
Project: Skin Lesion Similarity Learning (Siamese Network)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import numpy as np
from PIL import Image
import random, numpy as np
from tqdm import tqdm

def make_balanced_pairs(image_names, targets, loader_open_image, target_pairs=5000, seed=42):
    """
    Create a balanced set of positive and negative pairs.
    """
    random.seed(seed)
    np.random.seed(seed)

    benign = [n for n,t in zip(image_names, targets) if t==0]
    malignant = [n for n,t in zip(image_names, targets) if t==1]

    half = target_pairs // 2
    pos_pairs, neg_pairs = [], []

    # Positive pairs
    for _ in tqdm(range(half), desc="Making positive pairs"):
        if random.random() < len(benign) / max(1, len(benign)+len(malignant)):
            a,b = random.choices(benign, k=2)
        else:
            a,b = random.choices(malignant, k=2)
        img1 = loader_open_image(a)
        img2 = loader_open_image(b)
        if img1 is not None and img2 is not None:
            pos_pairs.append((img1, img2, 1))

    # Negative pairs
    for _ in tqdm(range(half), desc="Making negative pairs"):
        a = random.choice(benign)
        b = random.choice(malignant)
        img1 = loader_open_image(a)
        img2 = loader_open_image(b)
        if img1 is not None and img2 is not None:
            neg_pairs.append((img1, img2, 0))

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)

    x1 = np.array([p[0] for p in all_pairs])
    x2 = np.array([p[1] for p in all_pairs])
    y  = np.array([p[2] for p in all_pairs])

    return x1, x2, y

class SiamesePairDataset(Dataset):
    """Dataset wrapper for pickle-based Siamese pairs."""
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Handles both [[x1, x2], y] and full split pickles
        if isinstance(data, list) and len(data) == 2:
            (self.x1, self.x2), self.y = data
        elif isinstance(data, list) and len(data) == 4:
            (self.x1, self.x2), self.y, _, _ = data
        else:
            raise ValueError(f"Unexpected pickle format: {type(data)}")

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img1 = Image.fromarray((self.x1[idx] * 255).astype(np.uint8))
        img2 = Image.fromarray((self.x2[idx] * 255).astype(np.uint8))
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, label
    def _make_pairs(self, target_pairs=5000):
      """
      Uses make_balanced_pairs() to create a balanced dataset of pairs.
      """
      image_names = self.df['image_name'].values
      targets = self.df['target'].values
      x1, x2, y = make_balanced_pairs(
          image_names, targets, self._open_image, target_pairs=target_pairs
      )
      return x1, x2, y



def create_data_loaders(train_pkl, val_pkl=None, batch_size=16):
    """Convenience function for creating train/val dataloaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = SiamesePairDataset(train_pkl, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_pkl:
        val_ds = SiamesePairDataset(val_pkl, transform)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"✅ Loaded: {len(train_ds)} training pairs")
    if val_loader:
        print(f"✅ Loaded: {len(val_loader.dataset)} validation pairs")

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader = create_data_loaders(
        "/content/drive/MyDrive/siamese_isic_split.pkl"
    )
    img1, img2, y = next(iter(train_loader))
    print("Batch shapes:", img1.shape, img2.shape, y.shape)
