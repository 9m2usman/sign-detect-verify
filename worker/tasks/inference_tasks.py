"""
Celery Asynchronous Inference Tasks
Handles decoupled document processing, signature cropping, similarity calculation,
and database persistence outside the FastAPI HTTP event loop.
"""

import os
from datetime import datetime
from worker.celery_app import celery_app
from backend.app.database.session import SessionLocal
from backend.app.models.job import Job, VerificationResult

_pipeline = None


def get_pipeline():
    """Lazy loader for inference pipeline to ensure instantaneous app startup."""
    global _pipeline
    if _pipeline is None:
        from mlops.pipeline import UnifiedSignaturePipeline
        _pipeline = UnifiedSignaturePipeline()
    return _pipeline


@celery_app.task(bind=True, name="worker.tasks.inference_tasks.process_verification_job", max_retries=3)
def process_verification_job(self, job_id: str = None):
    """
    Asynchronous task:
    1. Fetches job from DB.
    2. Runs UnifiedSignaturePipeline on document and reference signatures.
    3. Persists results and updates job state to COMPLETED.
    """
    # Handle direct function call without Celery worker
    if job_id is None and isinstance(self, str):
        job_id = self

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"status": "error", "message": f"Job {job_id} not found."}

        job.status = "PROCESSING"
        db.commit()

        # Run pipeline
        storage_dir = os.path.dirname(job.document_path)
        crops_dir = os.path.join(storage_dir, f"crops_{job_id}")

        pipeline = get_pipeline()
        pipeline_output = pipeline.process_document(
            document_path=job.document_path,
            reference_paths=job.reference_paths,
            save_crops_dir=crops_dir
        )

        job.detected_count = pipeline_output["detected_signatures_count"]

        # Save individual verification results
        for item in pipeline_output["results"]:
            res = VerificationResult(
                job_id=job.id,
                signature_index=item["signature_index"],
                bbox=item["bbox"],
                confidence_detection=item["confidence_detection"],
                reference_path=item["best_reference"],
                similarity_score=item["similarity_score"],
                decision=item["decision"],
                crop_path=item["crop_path"]
            )
            db.add(res)

        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow()
        db.commit()

        return {
            "status": "success",
            "job_id": job_id,
            "detected_count": job.detected_count
        }

    except Exception as exc:
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
            db.commit()
        # Retry with exponential backoff if celery retry available
        if hasattr(self, "retry"):
            raise self.retry(exc=exc, countdown=5)
        raise exc
    finally:
        db.close()
