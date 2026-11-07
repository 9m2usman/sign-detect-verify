"""
RT-DETR Remote Training Script for Signature Detection
Architecture: RT-DETR-R50 (Real-Time DEtection TRansformer)
License: Apache 2.0 (Baidu / Hugging Face Transformers)
Dataset: 1-class (Signature)
Target Platform: Google Colab / Kaggle GPU / AWS EC2 GPU
Experiment Tracking: Comet ML & MLflow
"""

import os
import argparse
import json
import torch
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
class RTDETRConfig:
    exp_name: str = "rtdetr_r50_signature_detection"
    model_id: str = "PekingU/rtdetr_r50vd"
    dataset_dir: str = "./merged_signature_dataset"
    output_dir: str = "./checkpoints/rtdetr_r50"
    num_classes: int = 1
    input_size: tuple = (640, 640)
    batch_size: int = 8
    max_epochs: int = 30
    learning_rate: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    tracker: str = "comet"


def setup_experiment_tracking(config: RTDETRConfig):
    tracker_instance = None
    if config.tracker == "comet" and COMET_AVAILABLE:
        api_key = os.getenv("COMET_API_KEY")
        project_name = os.getenv("COMET_PROJECT_NAME", "signature-detection-rtdetr")
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
    parser = argparse.ArgumentParser(description="Train RT-DETR-R50 on Signature Dataset")
    parser.add_argument("--dataset-dir", type=str, default="./merged_signature_dataset")
    parser.add_argument("--model-id", type=str, default="PekingU/rtdetr_r50vd")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/rtdetr_r50")
    parser.add_argument("--tracker", type=str, choices=["comet", "mlflow", "none"], default="comet")
    return parser.parse_args()


def main():
    args = parse_args()
    config = RTDETRConfig(
        dataset_dir=args.dataset_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
        tracker=args.tracker
    )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f"[*] Starting RT-DETR Signature Detection Training Pipeline")
    print(f"[*] Base Model: {config.model_id} (Apache 2.0)")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Dataset: {config.dataset_dir}")
    print(f"[*] Epochs: {config.max_epochs} | Batch Size: {config.batch_size}")
    print(f"[*] Checkpoint Target: {config.output_dir}")
    print("============================================================")

    tracker = setup_experiment_tracking(config)

    with open(os.path.join(config.output_dir, "train_config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    print("[+] Configuration recorded. Ready for remote GPU execution.")

if __name__ == "__main__":
    main()
