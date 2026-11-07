"""
Vision Transformer (ViT-B/16) Remote Training Script for Signature Verification
Objective: Deep Metric Learning on [CLS] Token Representations
License: Apache 2.0 (Hugging Face Transformers / Google)
Input: Cropped Signature Pairs (Genuine vs. Forged)
Platform: Google Colab / Kaggle GPU
Experiment Tracking: Comet ML & MLflow
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class ViTVerificationConfig:
    exp_name: str = "vit_b16_signature_verification"
    model_name: str = "google/vit-base-patch16-224-in21k"
    embedding_dim: int = 768
    batch_size: int = 16
    max_epochs: int = 25
    learning_rate: float = 3e-5
    margin: float = 0.5
    output_dir: str = "./checkpoints/vit_verifier"
    tracker: str = "comet"


def setup_experiment_tracking(config: ViTVerificationConfig):
    tracker_instance = None
    if config.tracker == "comet" and COMET_AVAILABLE:
        api_key = os.getenv("COMET_API_KEY")
        project_name = os.getenv("COMET_PROJECT_NAME", "signature-verification-vit")
        if api_key:
            tracker_instance = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                auto_metric_logging=True
            )
            tracker_instance.set_name(config.exp_name)
            tracker_instance.log_parameters(config.__dict__)
            print(f"[+] Comet ML initialized for project: {project_name}")
    elif config.tracker == "mlflow" and MLFLOW_AVAILABLE:
        mlflow.set_experiment(config.exp_name)
        mlflow.start_run()
        mlflow.log_params(config.__dict__)
        tracker_instance = "mlflow"
    return tracker_instance


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-B/16 for Signature Verification")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/vit_verifier")
    parser.add_argument("--tracker", type=str, choices=["comet", "mlflow", "none"], default="comet")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ViTVerificationConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        tracker=args.tracker
    )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f"[*] Starting ViT-B/16 Signature Verification Training Pipeline")
    print(f"[*] Base Model: {config.model_name} (768-dim [CLS] token)")
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
