"""
Signatures API Endpoints: Document Verification Job Submission.
"""

import os
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.job import Job
from backend.app.schemas.jobs import JobCreateResponse, JobStatusResponse, SignatureVerificationResultItem
from backend.app.core.config import settings
from worker.tasks.inference_tasks import process_verification_job

router = APIRouter()


@router.post(
    "/verify",
    response_model=None,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit document and reference signatures for asynchronous or synchronous verification"
)
def verify_signature_endpoint(
    document: UploadFile = File(..., description="Document image containing signatures to verify"),
    references: List[UploadFile] = File(default=[], description="One or more authentic reference signatures"),
    reference_signatures: List[UploadFile] = File(default=[], description="Alternative field name for references"),
    sync: bool = Query(False, description="Whether to execute synchronously and return results immediately"),
    db: Session = Depends(get_db)
):
    """
    Submits a document and reference signatures.
    Creates a background verification job and publishes it to RabbitMQ/Celery.
    If sync=True, waits for execution and immediately returns final results.
    """
    if not document.filename:
        raise HTTPException(status_code=400, detail="Missing document file.")

    # Accept either 'references' or 'reference_signatures' field names
    ref_list = references if isinstance(references, list) else ([references] if references else [])
    alt_list = reference_signatures if isinstance(reference_signatures, list) else ([reference_signatures] if reference_signatures else [])
    valid_refs = [r for r in (ref_list + alt_list) if hasattr(r, "filename") and r.filename]
    if not valid_refs:
        raise HTTPException(status_code=400, detail="At least one reference signature is required.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save document
    doc_path = os.path.join(job_dir, f"doc_{document.filename}")
    with open(doc_path, "wb") as f:
        f.write(document.file.read())

    # Save references
    ref_paths = []
    ref_filenames = []
    for idx, ref in enumerate(valid_refs):
        ref_path = os.path.join(job_dir, f"ref_{idx}_{ref.filename}")
        with open(ref_path, "wb") as f:
            f.write(ref.file.read())
        ref_paths.append(ref_path)
        ref_filenames.append(ref.filename)

    from backend.app.api.v1.endpoints.jobs import to_web_url

    # Create persistent DB record
    job = Job(
        id=job_id,
        status="PENDING",
        document_path=doc_path,
        document_url=to_web_url(doc_path),
        reference_paths=ref_paths,
        detected_count=0
    )
    db.add(job)
    db.commit()

    # If synchronous mode requested (or running local fallback without broker)
    if sync:
        process_verification_job(job_id)
        # Fetch updated job and results from DB
        db.refresh(job)
        from backend.app.api.v1.endpoints.jobs import to_web_url
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

    # Otherwise dispatch to Celery queue (or run locally if broker not running)
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
