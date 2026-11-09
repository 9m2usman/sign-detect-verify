"""
Pydantic v2 Schemas for Requests, Responses, and Webhooks.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SignatureVerificationResultItem(BaseModel):
    """Schema for individual verified signature region."""
    signature_index: int
    bbox: List[float] = Field(..., description="[xmin, ymin, xmax, ymax] coordinates")
    confidence_detection: float
    reference_filename: str
    similarity_score: float
    decision: str = Field(..., description="GENUINE, FORGED, or REVIEW")
    crop_url: Optional[str] = None


class JobCreateResponse(BaseModel):
    """Returned upon successful submission of verification job."""
    job_id: str
    status: str
    message: str
    document_filename: str
    reference_filenames: List[str]
    poll_url: str


class JobStatusResponse(BaseModel):
    """Returned when polling job processing status."""
    job_id: str
    status: str
    detected_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: List[SignatureVerificationResultItem] = []
    error_message: Optional[str] = None
    document_url: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Metadata describing active production models and licenses."""
    detector_model: str
    detector_license: str
    detector_parameters: str
    verifier_model: str
    verifier_license: str
    verifier_parameters: str
    threshold_genuine: float
    threshold_forged: float
    compliance_notice: str


class HuggingFaceWebhookPayload(BaseModel):
    """Payload sent by Hugging Face Hub dataset webhooks."""
    event: str = Field(..., description="Event type, e.g. 'update'")
    repo: Dict[str, Any] = Field(..., description="Repository info (name, type, author)")
    commit: Optional[Dict[str, Any]] = None
