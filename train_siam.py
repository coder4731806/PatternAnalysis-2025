import torch, pickle, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model_siam import SiameseNetwork

# -----------------------
# Dataset loader
# -----------------------
class PairDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, 'rb') as f:
            (self.x1, self.x2), self.y = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img1, img2, label = self.x1[i], self.x2[i], self.y[i]
        img1 = Image.fromarray((img1 * 255).astype(np.uint8))
        img2 = Image.fromarray((img2 * 255).astype(np.uint8))
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# -----------------------
# Simple Focal Loss
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

# -----------------------
# Training / Validation
# -----------------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        out, _, _ = model(x1, x2)
        loss = loss_fn(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(out.squeeze()) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out, _, _ = model(x1, x2)
            loss = loss_fn(out.squeeze(), y)
            total_loss += loss.item()
            preds = (torch.sigmoid(out.squeeze()) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# -----------------------
# Main
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = PairDataset("/content/drive/MyDrive/siam2/siamese_isic_train.pkl", transform)
    val_ds   = PairDataset("/content/drive/MyDrive/siam2/siamese_isic_val.pkl", transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = FocalLoss()

    best_acc = 0
    for epoch in range(1, 16):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, Time={time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/siam2/siamese_best.pth")
            print(f"💾 Model saved at epoch {epoch}")

    print("Training complete. Best validation accuracy:", best_acc)

if __name__ == "__main__":
    main()
