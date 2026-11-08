"""
MLflow Experiment Tracking and Metrics Logging Helper.
Records hyperparameters, training loss curves, mAP@50, mAP@50:95, EER, and saves model checkpoints.
Provides local filesystem fallback when mlflow package is not installed.
"""

import os
import json
import uuid
from typing import Dict, Any, Optional
from backend.app.core.config import settings

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTracker:
    """
    Manages experiment logging to an MLflow Tracking Server or local metadata store.
    """
    def __init__(self, experiment_name: str = None, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        self.is_connected = False

        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.is_connected = True
            except Exception:
                try:
                    mlflow.set_tracking_uri("file:./mlruns")
                    mlflow.set_experiment(self.experiment_name)
                except Exception:
                    pass

    def log_training_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifact_path: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Logs a full model training and evaluation run.
        """
        if HAS_MLFLOW and self.is_connected:
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(parameters)
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
                if tags:
                    mlflow.set_tags(tags)
                if artifact_path and os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path="model_weights")
                return run.info.run_id
        else:
            # Local fallback run logger
            run_id = f"local_{uuid.uuid4().hex[:12]}"
            log_dir = os.path.join("mlruns", "local_experiments", run_id)
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": run_id,
                    "run_name": run_name,
                    "parameters": parameters,
                    "metrics": metrics,
                    "tags": tags or {}
                }, f, indent=2)
            return run_id
