from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBoxModel(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class CandidateMatch(BaseModel):
    candidate_index: int
    box_xyxy: List[int]
    detection_confidence: float
    similarity_score: float
    matched_reference_index: Optional[int]
    is_match: bool
    decision: str

class VerificationJobSubmissionResponse(BaseModel):
    job_id: str
    status: str = "QUEUED"
    message: str
    submitted_at: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # QUEUED, PROCESSING, COMPLETED, FAILED
    decision: Optional[str] = None
    similarity_score: Optional[float] = None
    is_match: Optional[bool] = None
    detections_count: Optional[int] = None
    matches: Optional[List[CandidateMatch]] = None
    error_message: Optional[str] = None
    processing_time_sec: Optional[float] = None
