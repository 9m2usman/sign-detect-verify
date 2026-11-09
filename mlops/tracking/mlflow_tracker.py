"""
MLflow Experiment Tracking and Model/Dataset Registry Helper.
Records hyperparameters, training loss curves, mAP@50, mAP@50:95, EER, and saves model checkpoints.
Provides first-class logging for:
1. Training Datasets (metadata, lineage, sample manifests, and splits)
2. PyTorch Model Artifacts (weights, architecture, and Model Registry registration)
Provides local filesystem fallback when mlflow tracking server is unreachable.
"""

import os
import json
import uuid
import socket
import logging
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def _is_server_reachable(url: str, timeout: float = 0.4) -> bool:
    """Fast non-blocking check whether an HTTP tracking server is online."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class MLflowTracker:
    """
    Manages experiment logging to an MLflow Tracking Server, SQLite store, or local metadata store.
    Supports logging datasets and PyTorch models to MLflow Model Registry.
    """
    def __init__(self, experiment_name: str = None, tracking_uri: str = None):
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.is_connected = False
        self.active_backend = "offline"

        if HAS_MLFLOW:
            candidates = []
            # 1. Try HTTP tracking URI only if actively listening (avoids long socket timeout hangs)
            if self.tracking_uri and (self.tracking_uri.startswith("http://") or self.tracking_uri.startswith("https://")):
                if _is_server_reachable(self.tracking_uri):
                    candidates.append(self.tracking_uri)
            elif self.tracking_uri and self.tracking_uri.startswith("sqlite"):
                candidates.append(self.tracking_uri)

            # 2. SQLite local database for instant zero-latency persistent runs & model registry
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_path = os.path.join(project_root, "mlflow.db").replace("\\", "/")
            candidates.append(f"sqlite:///{db_path}")
            # 3. File fallback
            candidates.append(f"file:{os.path.join(project_root, 'mlruns').replace('\\', '/')}")

            for uri in candidates:
                try:
                    mlflow.set_tracking_uri(uri)
                    mlflow.set_experiment(self.experiment_name)
                    self.is_connected = True
                    self.active_backend = uri
                    logger.info(f"MLflow connected to backend: {uri}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to connect to MLflow URI {uri}: {e}")
                    continue

    def log_dataset_manifest(self, dataset_info: Dict[str, Any]) -> None:
        """
        Logs dataset lineage, metadata, and manifest dictionary to the active MLflow run.
        """
        if not HAS_MLFLOW or not self.is_connected:
            return

        try:
            # Flatten high-level scalar parameters for easy filtering in MLflow UI
            dataset_params = {}
            for k, v in dataset_info.items():
                if isinstance(v, (str, int, float, bool)):
                    dataset_params[f"dataset.{k}"] = str(v)
            if dataset_params:
                mlflow.log_params(dataset_params)

            # Log rich structured metadata artifact
            mlflow.log_dict(dataset_info, "dataset/dataset_manifest.json")
        except Exception as e:
            logger.warning(f"Error logging dataset to MLflow: {e}")

    def log_pytorch_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Logs a PyTorch model into MLflow with architecture and registers it in the Model Registry.
        """
        if not HAS_MLFLOW or not self.is_connected or model is None:
            return None

        try:
            try:
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name=artifact_path,
                    registered_model_name=registered_model_name
                )
            except (TypeError, ValueError):
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
            return getattr(model_info, "model_uri", None)
        except Exception as e:
            logger.warning(f"Error logging PyTorch model to MLflow: {e}")
            return None

    def log_training_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifact_path: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        Logs a full model training and evaluation run, including:
        - Hyperparameters & training configuration
        - Evaluation metrics (mAP, EER, Loss, Latency)
        - Dataset metadata and manifest
        - PyTorch model checkpoint (.pth) and MLflow model artifact
        - MLflow Model Registry promotion if registered_model_name is provided
        """
        if HAS_MLFLOW and self.is_connected:
            try:
                with mlflow.start_run(run_name=run_name) as run:
                    # 1. Log parameters
                    clean_params = {}
                    for k, v in parameters.items():
                        clean_params[str(k)] = str(v) if not isinstance(v, (int, float, bool)) else v
                    mlflow.log_params(clean_params)

                    # 2. Log metrics
                    for k, v in metrics.items():
                        try:
                            mlflow.log_metric(k, float(v))
                        except (ValueError, TypeError):
                            pass

                    # 3. Log tags
                    combined_tags = tags.copy() if tags else {}
                    combined_tags["framework"] = "PyTorch"
                    combined_tags["license"] = "Permissive (Apache-2.0 / BSD-3)"
                    combined_tags["zero_agpl"] = "true"
                    mlflow.set_tags(combined_tags)

                    # 4. Log Dataset Information to MLflow
                    if dataset_info:
                        self.log_dataset_manifest(dataset_info)

                    # 5. Log Model Checkpoint File (.pth)
                    if artifact_path and os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, artifact_path="checkpoints")

                    # 6. Log PyTorch Model to MLflow & Model Registry
                    if model is not None:
                        self.log_pytorch_model(
                            model=model,
                            artifact_path="model",
                            registered_model_name=registered_model_name
                        )

                    return run.info.run_id
            except Exception as e:
                logger.warning(f"MLflow run logging failed with exception: {e}. Falling back to local logger.")

        # Local fallback run logger when MLflow tracking server is completely unreachable
        import tempfile
        run_id = f"local_{uuid.uuid4().hex[:12]}"
        try:
            log_dir = os.path.join("mlruns", "local_experiments", run_id)
            os.makedirs(log_dir, exist_ok=True)
        except (PermissionError, OSError):
            log_dir = os.path.join(tempfile.gettempdir(), "mlruns", "local_experiments", run_id)
            os.makedirs(log_dir, exist_ok=True)

        record = {
            "run_id": run_id,
            "run_name": run_name,
            "parameters": parameters,
            "metrics": metrics,
            "tags": tags or {},
            "dataset_info": dataset_info or {},
            "artifact_path": artifact_path,
            "registered_model_name": registered_model_name
        }
        with open(os.path.join(log_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return run_id
