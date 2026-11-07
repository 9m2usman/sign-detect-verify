"""
YOLOX-S Remote Training Script for Signature Detection
Dataset: YOLO-format 1-class (Signature)
Target Platform: Google Colab / Kaggle GPU / AWS EC2 GPU
License: Apache 2.0 (Megvii YOLOX)
Experiment Tracking: Comet ML & MLflow
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.distributed as dist
from dataclasses import dataclass

# Ensure Comet ML is imported first if enabled
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
class TrainingConfig:
    exp_name: str = "yolox_s_signature_detection"
    dataset_dir: str = "./merged_signature_dataset"
    output_dir: str = "./checkpoints/yolox_s"
    num_classes: int = 1
    depth: float = 0.33      # YOLOX-S depth multiplier
    width: float = 0.50      # YOLOX-S width multiplier
    input_size: tuple = (640, 640)
    batch_size: int = 16
    max_epoch: int = 30
    warmup_epochs: int = 3
    basic_lr_per_img: float = 0.01 / 64.0
    weight_decay: float = 5e-4
    momentum: float = 0.9
    save_history_ckpt: bool = True
    eval_interval: int = 1
    seed: int = 42
    tracker: str = "comet"  # 'comet', 'mlflow', or 'none'


def setup_experiment_tracking(config: TrainingConfig):
    tracker_instance = None
    if config.tracker == "comet" and COMET_AVAILABLE:
        api_key = os.getenv("COMET_API_KEY")
        project_name = os.getenv("COMET_PROJECT_NAME", "signature-detection-yolox")
        if api_key:
            tracker_instance = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                auto_metric_logging=True
            )
            tracker_instance.set_name(config.exp_name)
            tracker_instance.log_parameters(config.__dict__)
            print(f"[+] Comet ML initialized for project: {project_name}")
        else:
            print("[!] COMET_API_KEY not set. Continuing without Comet ML.")
    elif config.tracker == "mlflow" and MLFLOW_AVAILABLE:
        mlflow.set_experiment(config.exp_name)
        mlflow.start_run()
        mlflow.log_params(config.__dict__)
        print(f"[+] MLflow tracking initialized for experiment: {config.exp_name}")
        tracker_instance = "mlflow"
    return tracker_instance


def log_epoch_metrics(tracker, epoch, train_loss, val_loss, map50, map50_95, lr):
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mAP@50": map50,
        "mAP@50:95": map50_95,
        "learning_rate": lr
    }
    if tracker and tracker != "mlflow":
        tracker.log_metrics(metrics, step=epoch)
    elif tracker == "mlflow":
        mlflow.log_metrics(metrics, step=epoch)
    print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mAP@50: {map50:.4f} | mAP@50:95: {map50_95:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOX-S on Signature Dataset")
    parser.add_argument("--dataset-dir", type=str, default="./merged_signature_dataset", help="Path to merged signature dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/yolox_s", help="Directory to save checkpoints")
    parser.add_argument("--tracker", type=str, choices=["comet", "mlflow", "none"], default="comet")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        max_epoch=args.epochs,
        output_dir=args.output_dir,
        tracker=args.tracker
    )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"[*] Starting YOLOX-S Signature Detection Training Pipeline")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Dataset: {config.dataset_dir}")
    print(f"[*] Epochs: {config.max_epoch} | Batch Size: {config.batch_size}")
    print(f"[*] Checkpoint Target: {config.output_dir}")
    print(f"============================================================")

    tracker = setup_experiment_tracking(config)

    # Save training metadata config
    with open(os.path.join(config.output_dir, "train_config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    print("[+] Configuration recorded. Ready for remote GPU execution.")

if __name__ == "__main__":
    main()
