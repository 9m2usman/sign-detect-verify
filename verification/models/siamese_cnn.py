"""
Siamese Convolutional Neural Network for Signature Verification
Utilizes twin weight-sharing deep convolutional branches to project signature images
into a normalized metric embedding space, trained using Contrastive Loss.
License: BSD-3-Clause / Apache 2.0 (Zero Ultralytics, Zero AGPL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseSignatureCNN(nn.Module):
    """
    Siamese CNN architecture with shared backbone encoder for pairwise signature verification.
    """
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Use permissive lightweight ResNet-18 or MobileNetV3 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Extract feature representation before final classifier
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection head with L2 normalization
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized embedding for a single signature image."""
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """Passes pair of signature images through shared twin encoder."""
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        return emb1, emb2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function:
    L = 0.5 * Y * d^2 + 0.5 * (1 - Y) * max(0, margin - d)^2
    where Y=1 for genuine pairs (similar) and Y=0 for forged/dissimilar pairs.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        loss_genuine = label * torch.pow(euclidean_distance, 2)
        loss_forged = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_genuine + loss_forged)
