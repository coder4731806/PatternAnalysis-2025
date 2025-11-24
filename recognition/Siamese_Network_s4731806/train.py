import torch, pickle, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from recognition.Siamese_Network_s4731806.modules import SiameseNetwork

# -----------------------
# Dataset loader
# -----------------------
class PairDataset(Dataset):
    """Simple dataset wrapper for precomputed Siamese pairs stored in pickle."""
    def __init__(self, pkl_path, transform=None):
        # Load the precomputed (x1, x2, y) pairs
        with open(pkl_path, 'rb') as f:
            (self.x1, self.x2), self.y = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Convert images from normalized arrays to PIL images
        img1, img2, label = self.x1[i], self.x2[i], self.y[i]
        img1 = Image.fromarray((img1 * 255).astype(np.uint8))
        img2 = Image.fromarray((img2 * 255).astype(np.uint8))
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        # Return images and label as float tensor
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# -----------------------
# Focal Loss
# -----------------------
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance.
    alpha -> weighting factor for positives
    gamma -> focusing parameter for hard examples
    """
    def __init__(self, alpha=0.7, gamma=1.5):
        super().__init__()
        self.a = alpha
        self.g = gamma

    def forward(self, x, y):
        # Compute BCE first, then focal modulation
        bce = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        pt = torch.exp(-bce)
        return (self.a * (1 - pt) ** self.g * bce).mean()


# -----------------------
# Training / Validation loops
# -----------------------
def train_epoch(model, loader, opt, loss_fn, device):
    """Run one training epoch over the dataset."""
    model.train()
    run_loss, correct, total = 0, 0, 0

    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        opt.zero_grad()
        out, _, _ = model(x1, x2)  # Siamese network returns distance/logits
        loss = loss_fn(out.squeeze(), y)
        loss.backward()
        opt.step()

        run_loss += loss.item()
        preds = (torch.sigmoid(out.squeeze()) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

    return run_loss / len(loader), correct / total


def validate(model, loader, loss_fn, device):
    """Evaluate model on validation set without updating weights."""
    model.eval()
    run_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out, _, _ = model(x1, x2)
            loss = loss_fn(out.squeeze(), y)
            run_loss += loss.item()
            preds = (torch.sigmoid(out.squeeze()) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    return run_loss / len(loader), correct / total


# -----------------------
# Main training loop
# -----------------------
def main():
    # Paths to your precomputed train/val pairs
    train_pkl = "/content/drive/MyDrive/siam2/siamese_isic_train.pkl"
    val_pkl   = "/content/drive/MyDrive/siam2/siamese_isic_val.pkl"

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard transform for Siamese network
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_ds = PairDataset(train_pkl, transform)
    val_ds = PairDataset(val_pkl, transform)

    # Wrap datasets in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Initialize model, optimizer, and loss
    model = SiameseNetwork().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = FocalLoss()

    best = 0
    for epoch in range(1, 16):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # Print epoch stats
        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")

        # Save model if validation accuracy improved
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/siam2/siamese_best.pth")
            print(f"💾 Model saved at epoch {epoch}")

        print(f"⏱ {time.time() - t0:.1f}s\n")

    print(f"✅ Training complete! Best val accuracy: {best:.3f}")


if __name__ == "__main__":
    main()
