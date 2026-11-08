"""
Production Signature Verifier (Siamese CNN & Triplet Metric Learning)
Extracts L2-normalized 256-dimensional embeddings from signature crops, computes cosine similarity
against one or more reference signatures, and applies calibrated decision thresholds.
Includes offline fallback for air-gapped test environments.
"""

import os
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from verification.models.siamese_cnn import SiameseSignatureCNN
from verification.preprocessing.preprocess import signature_to_tensor
from backend.app.core.config import settings


class ProductionSignatureVerifier:
    """
    Inference engine for signature verification.
    Computes pairwise and multi-reference cosine similarities.
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        genuine_threshold: float = None,
        forged_threshold: float = None,
        pretrained: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseSignatureCNN(embedding_dim=embedding_dim, pretrained=pretrained).to(self.device)
        self.model.eval()

        self.threshold_genuine = genuine_threshold or settings.THRESHOLD_GENUINE
        self.threshold_forged = forged_threshold or settings.THRESHOLD_FORGED

    @torch.no_grad()
    def extract_embedding(self, image: Union[Image.Image, str]) -> np.ndarray:
        """Extracts 256-dim L2-normalized embedding vector."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        tensor = signature_to_tensor(image).unsqueeze(0).to(self.device)
        emb = self.model.forward_one(tensor)
        return emb.cpu().squeeze(0).numpy()

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes Cosine Similarity between two unit-normalized vectors: S_C = u . v"""
        score = float(np.dot(emb1, emb2))
        return max(-1.0, min(1.0, score))

    def verify_against_references(
        self,
        crop_image: Union[Image.Image, str],
        reference_images: List[Union[Image.Image, str]]
    ) -> Dict[str, Any]:
        """
        Compares a detected signature crop against one or more reference signatures.
        Aggregates multiple references using calibrated maximum similarity.
        """
        if not reference_images:
            raise ValueError("At least one reference signature must be provided.")

        crop_emb = self.extract_embedding(crop_image)
        comparison_scores = []

        for ref in reference_images:
            ref_emb = self.extract_embedding(ref)
            sim = self.compute_similarity(crop_emb, ref_emb)
            comparison_scores.append(sim)

        aggregated_score = max(comparison_scores)
        best_ref_idx = int(np.argmax(comparison_scores))

        if aggregated_score >= self.threshold_genuine:
            decision = "GENUINE"
        elif aggregated_score <= self.threshold_forged:
            decision = "FORGED"
        else:
            decision = "REVIEW"

        return {
            "aggregated_similarity": round(aggregated_score, 4),
            "decision": decision,
            "best_reference_index": best_ref_idx,
            "individual_scores": [round(s, 4) for s in comparison_scores],
            "threshold_genuine": self.threshold_genuine,
            "threshold_forged": self.threshold_forged
        }
