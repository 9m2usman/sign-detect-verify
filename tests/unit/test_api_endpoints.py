"""
Unit Tests for FastAPI REST Endpoints.
Tests Health, Root, Models Info, Webhooks, and Error Status Codes.
"""

import pytest
from fastapi import HTTPException
from backend.app.main import app, health_check, root
from backend.app.api.v1.endpoints.models import get_model_info
from backend.app.api.v1.endpoints.webhooks import huggingface_webhook_receiver
from backend.app.schemas.jobs import HuggingFaceWebhookPayload
from backend.app.core.config import settings


def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert "active_detector" in res


def test_root_endpoint():
    res = root()
    assert "Welcome" in res["message"]
    assert res["docs_url"] == "/docs"


def test_models_info_endpoint():
    res = get_model_info()
    assert "RTMDet" in res.detector_model
    assert "Siamese" in res.verifier_model
    assert "Apache" in res.detector_license
    assert res.threshold_genuine > res.threshold_forged


def test_huggingface_webhook_unauthorized():
    payload = HuggingFaceWebhookPayload(
        event="update",
        repo={"name": "test-dataset", "type": "dataset"}
    )
    with pytest.raises(HTTPException) as exc_info:
        huggingface_webhook_receiver(payload, x_webhook_secret="invalid-secret")
    assert exc_info.value.status_code == 401


def test_huggingface_webhook_authorized():
    payload = HuggingFaceWebhookPayload(
        event="update",
        repo={"name": "user/signature-detection-dataset", "type": "dataset"}
    )
    res = huggingface_webhook_receiver(payload, x_webhook_secret=settings.HF_WEBHOOK_SECRET)
    assert res["status"] == "triggered"
    assert "detection" in res["retraining_target"]
