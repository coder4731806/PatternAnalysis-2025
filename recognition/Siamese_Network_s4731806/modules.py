import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    """ResNet18-based Siamese Network for Melanoma Similarity Learning."""
    def __init__(self, embedding_dim=128, dropout=0.3, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        base.fc = nn.Identity()
        self.backbone = base

        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.compare = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward_single(self, x):
        f = self.backbone(x)
        return self.embedding(f)

    def forward(self, x1, x2):
        e1, e2 = self.forward_single(x1), self.forward_single(x2)
        sim = self.compare(torch.cat([e1, e2], dim=1))
        return sim, e1, e2
