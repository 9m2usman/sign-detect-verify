"""
Jobs API Endpoints: Poll Status and Results.
"""

import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.job import Job
from backend.app.schemas.jobs import JobStatusResponse, SignatureVerificationResultItem

router = APIRouter()


import os
from typing import Optional
from backend.app.core.config import settings

def to_web_url(path: Optional[str]) -> Optional[str]:
    """Converts a local filesystem storage path to an HTTP static web URL."""
    if not path:
        return None
    norm = path.replace("\\", "/")
    if norm.startswith("/uploads/") or norm.startswith("/sample_test_run/") or norm.startswith("/signature_crops/"):
        return norm
    try:
        rel = os.path.relpath(path, settings.UPLOAD_DIR).replace("\\", "/")
        if not rel.startswith(".."):
            return f"/uploads/{rel}"
    except Exception:
        pass
    if "/uploads/" in norm:
        return norm[norm.find("/uploads/"):]
    if "signature_uploads/" in norm:
        return "/uploads/" + norm.split("signature_uploads/", 1)[1]
    if "signature_crops/" in norm:
        return "/signature_crops/" + norm.split("signature_crops/", 1)[1]
    if norm.startswith("uploads/"):
        return "/" + norm
    if "/sample_test_run/" in norm:
        return norm[norm.find("/sample_test_run/"):]
    return f"/uploads/{os.path.basename(norm)}"


@router.get(
    "/",
    response_model=List[JobStatusResponse],
    summary="List recent verification jobs"
)
def list_recent_jobs(limit: int = 20, db: Session = Depends(get_db)):
    """
    Returns the most recent verification jobs from the database for audit and dashboard monitoring.
    """
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    output = []
    for job in jobs:
        results_data = [
            SignatureVerificationResultItem(
                signature_index=res.signature_index,
                bbox=res.bbox,
                confidence_detection=res.confidence_detection,
                reference_filename=to_web_url(res.reference_path) or res.reference_path,
                similarity_score=res.similarity_score,
                decision=res.decision,
                crop_url=to_web_url(res.crop_path)
            ) for res in job.results
        ]
        output.append(
            JobStatusResponse(
                job_id=job.id,
                status=job.status,
                detected_count=job.detected_count,
                created_at=job.created_at,
                completed_at=job.completed_at,
                results=results_data,
                error_message=job.error_message,
                document_url=to_web_url(job.document_path)
            )
        )
    return output



@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Retrieve processing status and verification results for a job"
)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Polls the status of an asynchronous verification job.
    Returns detected signatures, bounding boxes, similarity scores, and decisions.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )

    results_data = []
    for res in job.results:
        results_data.append(
            SignatureVerificationResultItem(
                signature_index=res.signature_index,
                bbox=res.bbox,
                confidence_detection=res.confidence_detection,
                reference_filename=to_web_url(res.reference_path) or res.reference_path,
                similarity_score=res.similarity_score,
                decision=res.decision,
                crop_url=to_web_url(res.crop_path)
            )
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        detected_count=job.detected_count,
        created_at=job.created_at,
        completed_at=job.completed_at,
        results=results_data,
        error_message=job.error_message,
        document_url=to_web_url(job.document_path)
    )
