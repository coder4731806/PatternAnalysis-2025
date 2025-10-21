"""
dataset_siamese.py
Initial dataset loader for Siamese Network (PyTorch)
----------------------------------------------------
Loads .pkl files containing precomputed (x1, x2, y) pairs
for training or validation.

Author: Harshit Vishwa
Project: Skin Lesion Similarity Learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import numpy as np
from PIL import Image

class SiamesePairDataset(Dataset):
    #base implementation
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Expecting [[x1, x2], y]
        (self.x1, self.x2), self.y = data
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


def create_data_loaders(train_pkl, val_pkl=None, batch_size=16):
    """Basic function for creating train/val dataloaders."""
    #not good accuracy and long training times
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

    print(f"Loaded {len(train_ds)} training pairs")
    if val_loader:
        print(f"Loaded {len(val_loader.dataset)} validation pairs")

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader = create_data_loaders("/content/drive/MyDrive/siamese_isic_split.pkl")
    img1, img2, y = next(iter(train_loader))
    print("Batch shapes:", img1.shape, img2.shape, y.shape)
