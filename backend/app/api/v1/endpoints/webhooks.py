"""
Hugging Face Webhook Receiver: Automated Retraining Trigger.
"""

import hmac
import hashlib
from fastapi import APIRouter, Header, HTTPException, Request, status
from backend.app.schemas.jobs import HuggingFaceWebhookPayload
from backend.app.core.config import settings
from mlops.training_pipeline.retrain_orchestrator import RetrainingOrchestrator

router = APIRouter()
_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetrainingOrchestrator()
    return _orchestrator


@router.post(
    "/huggingface",
    status_code=status.HTTP_200_OK,
    summary="Receive Hugging Face dataset update events and trigger retraining"
)
def huggingface_webhook_receiver(
    payload: HuggingFaceWebhookPayload,
    x_webhook_secret: str = Header(None, alias="X-Webhook-Secret")
):
    """
    Validates webhook secret, inspects repo and event type, and schedules automated retraining.
    Distinguishes between detection dataset updates vs verification dataset updates.
    """
    # 1. Validate Secret Token
    if not x_webhook_secret or not hmac.compare_digest(x_webhook_secret.strip(), settings.HF_WEBHOOK_SECRET.strip()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Webhook-Secret header."
        )

    # 2. Inspect Repository Type & Event
    repo_name = payload.repo.get("name", "")
    event_type = payload.event

    if event_type not in ["update", "create", "push"]:
        return {
            "status": "ignored",
            "message": f"Event '{event_type}' ignored. Only dataset update/push triggers retraining."
        }

    # 3. Route to Appropriate Training Workflow
    if "detection" in repo_name.lower():
        retraining_target = "detection_rtmdet"
    elif "verification" in repo_name.lower() or "cedar" in repo_name.lower():
        retraining_target = "verification_siamese"
    else:
        retraining_target = "detection_rtmdet"

    # 4. Trigger Retraining Pipeline Orchestrator
    event_id = payload.commit.get("id", "event-manual") if payload.commit else f"event-{repo_name}"
    orchestrator = get_orchestrator()
    run_outcome = orchestrator.handle_dataset_update_event(
        event_id=event_id,
        payload=payload.model_dump()
    )

    return {
        "status": "triggered",
        "retraining_target": retraining_target,
        "repo": repo_name,
        "pipeline_outcome": run_outcome,
        "message": f"Automated remote GPU retraining job queued for target: {retraining_target}."
    }
