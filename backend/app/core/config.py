"""
Application Configuration Settings
Uses Pydantic BaseSettings to parse environment variables with safe defaults for local and cloud environments.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Signature Detection & Verification Platform"
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
    UPLOAD_DIR: str = "uploads"
    STORAGE_DIR: str = "storage"
    
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

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"


settings = Settings()
