"""
Jobs API Endpoints: Poll Status and Results.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.job import Job
from backend.app.schemas.jobs import JobStatusResponse, SignatureVerificationResultItem

router = APIRouter()


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
                reference_filename=res.reference_path,
                similarity_score=res.similarity_score,
                decision=res.decision,
                crop_url=res.crop_path
            )
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        detected_count=job.detected_count,
        created_at=job.created_at,
        completed_at=job.completed_at,
        results=results_data,
        error_message=job.error_message
    )
