import abc
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

class BaseSignatureEmbeddingModel(abc.ABC):
    """Abstract Base Class establishing common interface for signature verification backbones."""

    @abc.abstractmethod
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extracts an L2-normalized 1D feature embedding vector."""
        pass

    @abc.abstractmethod
    def get_embedding_dim(self) -> int:
        """Returns the dimensionality of the feature vector."""
        pass

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine similarity between two L2-normalized embeddings.
        Outputs a normalized score in range [0.0, 1.0].
        """
        dot_product = float(np.dot(emb1, emb2))
        norm1 = float(np.linalg.norm(emb1))
        norm2 = float(np.linalg.norm(emb2))
        cos_sim = dot_product / (norm1 * norm2 + 1e-9)
        # Rescale [-1.0, 1.0] to [0.0, 1.0]
        normalized_score = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
        return round(normalized_score, 4)


class ResNet50SignatureVerifier(BaseSignatureEmbeddingModel):
    """
    Verification Candidate 1: ResNet50 Deep Feature Extractor.
    Extracts 2048-dimensional feature representations from the penultimate average pooling layer.
    """
    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.DEFAULT if pretrained and weights_path is None else None
        
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()  # Drop classification head to extract 2048-dim features

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding_dim(self) -> int:
        return 2048

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        l2_normalized = F.normalize(features, p=2, dim=1)
        return l2_normalized.squeeze(0).cpu().numpy()


class ViTSignatureVerifier(BaseSignatureEmbeddingModel):
    """
    Verification Candidate 2: Vision Transformer (ViT-B/16).
    Extracts 768-dimensional token representation from the [CLS] token.
    Permissively licensed (Apache 2.0 via Hugging Face Transformers).
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", weights_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import ViTImageProcessor, ViTModel

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def get_embedding_dim(self) -> int:
        return 768

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Use [CLS] token output from the last hidden state
        cls_token = outputs.last_hidden_state[:, 0, :]
        l2_normalized = F.normalize(cls_token, p=2, dim=1)
        return l2_normalized.squeeze(0).cpu().numpy()
