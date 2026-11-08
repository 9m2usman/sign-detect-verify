"""
Unit Tests for Signature Verification Metric Learning Architectures
Tests Siamese twin forward pass, Triplet triplet forward pass, and loss functions.
"""

import pytest
import torch
from verification.models.siamese_cnn import SiameseSignatureCNN, ContrastiveLoss
from verification.models.triplet_network import TripletSignatureNetwork, TripletLoss


def test_siamese_cnn_forward():
    model = SiameseSignatureCNN(embedding_dim=128, pretrained=False)
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    
    emb1, emb2 = model(img1, img2)
    
    assert emb1.shape == (2, 128)
    assert emb2.shape == (2, 128)
    
    # Check L2 unit norm
    norm1 = torch.norm(emb1, p=2, dim=1)
    assert torch.allclose(norm1, torch.ones(2), atol=1e-5)


def test_contrastive_loss():
    loss_fn = ContrastiveLoss(margin=1.0)
    emb1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    emb2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([1.0, 0.0])  # Pair 1 genuine, Pair 2 forged
    
    loss = loss_fn(emb1, emb2, labels)
    assert loss.item() >= 0.0


def test_triplet_network_forward():
    model = TripletSignatureNetwork(embedding_dim=128, pretrained=False)
    a = torch.randn(2, 3, 224, 224)
    p = torch.randn(2, 3, 224, 224)
    n = torch.randn(2, 3, 224, 224)
    
    emb_a, emb_p, emb_n = model(a, p, n)
    
    assert emb_a.shape == (2, 128)
    assert emb_p.shape == (2, 128)
    assert emb_n.shape == (2, 128)


def test_triplet_loss():
    loss_fn = TripletLoss(margin=0.5)
    emb_a = torch.tensor([[1.0, 0.0]])
    emb_p = torch.tensor([[0.9, 0.1]])
    emb_n = torch.tensor([[0.0, 1.0]])
    
    loss = loss_fn(emb_a, emb_p, emb_n)
    assert loss.item() >= 0.0
