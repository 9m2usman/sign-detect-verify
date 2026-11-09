"""
Application Configuration Settings
Uses Pydantic v2 BaseSettings and ConfigDict to parse environment variables with safe defaults.
"""

import os
import tempfile
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


def _get_writable_dir(dir_name: str, fallback_prefix: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not os.path.isabs(dir_name):
        dir_name = os.path.join(project_root, dir_name)
    try:
        os.makedirs(dir_name, exist_ok=True)
        test_file = os.path.join(dir_name, f".test_{os.getpid()}")
        with open(test_file, "w") as f:
            f.write("1")
        os.remove(test_file)
        return os.path.abspath(dir_name)
    except Exception:
        fallback = os.path.join(tempfile.gettempdir(), fallback_prefix)
        os.makedirs(fallback, exist_ok=True)
        return os.path.abspath(fallback)


class Settings(BaseSettings):
    PROJECT_NAME: str = "Signature Detection & Verification API"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///./signature_dev.db"  # Fallback to local SQLite if Postgres not running
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "signature_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Message Broker (RabbitMQ & Celery)
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "rpc://"

    # Storage Paths
    UPLOAD_DIR: str = _get_writable_dir("uploads", "signature_uploads")
    STORAGE_DIR: str = _get_writable_dir("storage", "signature_storage")
    
    # MLOps & MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "signature-verification-prod"
    HF_WEBHOOK_SECRET: str = "dev-webhook-secret-token"

    # Model Configuration & Calibration Thresholds
    ACTIVE_DETECTOR: str = "rtmdet_tiny"
    ACTIVE_VERIFIER: str = "siamese_resnet18"
    THRESHOLD_GENUINE: float = 0.82
    THRESHOLD_FORGED: float = 0.65
    MAX_SIGNATURES_PER_DOC: int = 10

    model_config = ConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="allow"
    )


settings = Settings()
