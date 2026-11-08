"""
Signatures API Endpoints: Document Verification Job Submission.
"""

import os
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.job import Job
from backend.app.schemas.jobs import JobCreateResponse
from backend.app.core.config import settings
from worker.tasks.inference_tasks import process_verification_job

router = APIRouter()


@router.post(
    "/verify",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit document and reference signatures for asynchronous verification"
)
async def verify_signature_endpoint(
    document: UploadFile = File(..., description="Document image containing signatures to verify"),
    references: List[UploadFile] = File(..., description="One or more authentic reference signatures"),
    db: Session = Depends(get_db)
):
    """
    Submits a document and reference signatures.
    Creates a background verification job and publishes it to RabbitMQ/Celery.
    Returns 202 Accepted with polling URL.
    """
    if not document.filename:
        raise HTTPException(status_code=400, detail="Missing document file.")
    if not references:
        raise HTTPException(status_code=400, detail="At least one reference signature is required.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save document
    doc_path = os.path.join(job_dir, f"doc_{document.filename}")
    with open(doc_path, "wb") as f:
        f.write(await document.read())

    # Save references
    ref_paths = []
    ref_filenames = []
    for idx, ref in enumerate(references):
        ref_path = os.path.join(job_dir, f"ref_{idx}_{ref.filename}")
        with open(ref_path, "wb") as f:
            f.write(await ref.read())
        ref_paths.append(ref_path)
        ref_filenames.append(ref.filename)

    # Create persistent DB record
    job = Job(
        id=job_id,
        status="PENDING",
        document_path=doc_path,
        reference_paths=ref_paths,
        detected_count=0
    )
    db.add(job)
    db.commit()

    # Dispatch to Celery queue (or run locally if broker not running)
    try:
        process_verification_job.delay(job_id)
    except Exception:
        # Fallback to direct synchronous execution for local CPU testing without broker
        process_verification_job(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status="PENDING",
        message="Verification job successfully submitted and queued for processing.",
        document_filename=document.filename,
        reference_filenames=ref_filenames,
        poll_url=f"/api/v1/jobs/{job_id}"
    )
