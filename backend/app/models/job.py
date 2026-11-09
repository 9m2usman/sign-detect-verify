"""
SQLAlchemy ORM Models for Verification Jobs and Detected Signatures.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database.base import Base


class Job(Base):
    """Represents an asynchronous document verification job."""
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    document_path: Mapped[str] = mapped_column(String(500), nullable=False)
    document_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    reference_paths: Mapped[list] = mapped_column(JSON, default=list)
    detected_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    results: Mapped[List["VerificationResult"]] = relationship("VerificationResult", back_populates="job", cascade="all, delete-orphan")


class VerificationResult(Base):
    """Stores individual detected signature bounding box, similarity score, and authenticity decision."""
    __tablename__ = "verification_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    signature_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)  # [xmin, ymin, xmax, ymax]
    confidence_detection: Mapped[float] = mapped_column(Float, default=1.0)
    reference_path: Mapped[str] = mapped_column(String(500), nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # GENUINE, FORGED, REVIEW
    crop_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    job: Mapped["Job"] = relationship("Job", back_populates="results")


class ModelVersion(Base):
    """Tracks deployed model versions, metadata, and evaluation metrics."""
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'detection' or 'verification'
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), default="PyTorch")
    license: Mapped[str] = mapped_column(String(50), default="Apache-2.0")
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

