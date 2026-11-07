"""
ResNet50 Remote Training Script for Signature Verification
Objective: Deep Metric Learning (Contrastive / Triplet Loss / BCE on Cosine Similarity)
License: BSD 3-Clause (torchvision)
Input: Cropped Signature Pairs (Genuine vs. Forged / Same vs. Different Signer)
Platform: Google Colab / Kaggle GPU
Experiment Tracking: Comet ML & MLflow
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

@dataclass
class VerificationConfig:
    exp_name: str = "resnet50_signature_verification"
    embedding_dim: int = 2048
    batch_size: int = 32
    max_epochs: int = 25
    learning_rate: float = 1e-4
    margin: float = 0.5  # Contrastive margin
    output_dir: str = "./checkpoints/resnet50_verifier"
    tracker: str = "comet"


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Signature Verification.
    L(y, x1, x2) = (1 - y) * 0.5 * d^2 + y * 0.5 * max(0, margin - d)^2
    where y=0 indicates Genuine Pair, y=1 indicates Impostor/Forgery Pair.
    """
    def __init__(self, margin: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Euclidean distance between L2-normalized embeddings
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        loss = torch.mean(
            (1.0 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


def setup_experiment_tracking(config: VerificationConfig):
    tracker_instance = None
    if config.tracker == "comet" and COMET_AVAILABLE:
        api_key = os.getenv("COMET_API_KEY")
        project_name = os.getenv("COMET_PROJECT_NAME", "signature-verification-resnet50")
        if api_key:
            tracker_instance = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                auto_metric_logging=True
            )
            tracker_instance.set_name(config.exp_name)
            tracker_instance.log_parameters(config.__dict__)
            print(f"[+] Comet ML initialized for verification project: {project_name}")
    elif config.tracker == "mlflow" and MLFLOW_AVAILABLE:
        mlflow.set_experiment(config.exp_name)
        mlflow.start_run()
        mlflow.log_params(config.__dict__)
        tracker_instance = "mlflow"
    return tracker_instance


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Signature Verification")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/resnet50_verifier")
    parser.add_argument("--tracker", type=str, choices=["comet", "mlflow", "none"], default="comet")
    return parser.parse_args()


def main():
    args = parse_args()
    config = VerificationConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        tracker=args.tracker
    )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f"[*] Starting ResNet50 Metric Verification Training Pipeline")
    print(f"[*] Architecture: ResNet50 (2048-dim penultimate embedding)")
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Epochs: {config.max_epochs} | Batch Size: {config.batch_size}")
    print(f"[*] Checkpoint Target: {config.output_dir}")
    print("============================================================")

    tracker = setup_experiment_tracking(config)

    with open(os.path.join(config.output_dir, "train_config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    print("[+] Configuration recorded. Ready for remote GPU execution.")

if __name__ == "__main__":
    main()
