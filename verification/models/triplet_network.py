"""
Triplet Network for Signature Metric Learning
Uses Anchor-Positive-Negative triplet formulations with Triplet Margin Loss
to enforce tight genuine clustering while repelling skilled forgeries.
License: BSD-3-Clause / Apache 2.0 (Zero Ultralytics, Zero AGPL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TripletSignatureNetwork(nn.Module):
    """
    Triplet architecture sharing a deep convolutional encoder across Anchor, Positive, and Negative feeds.
    """
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts unit-length L2-normalized metric embedding."""
        feat = self.backbone(x)
        emb = self.projection_head(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        emb_a = self.extract_embedding(anchor)
        emb_p = self.extract_embedding(positive)
        emb_n = self.extract_embedding(negative)
        return emb_a, emb_p, emb_n


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss:
    L(A, P, N) = max(0, d(A, P) - d(A, N) + margin)
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a: torch.Tensor, emb_p: torch.Tensor, emb_n: torch.Tensor) -> torch.Tensor:
        dist_ap = F.pairwise_distance(emb_a, emb_p, p=2)
        dist_an = F.pairwise_distance(emb_a, emb_n, p=2)
        loss = F.relu(dist_ap - dist_an + self.margin)
        return torch.mean(loss)
