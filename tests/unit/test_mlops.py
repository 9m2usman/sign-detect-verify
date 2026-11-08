"""
Unit Tests for MLOps Tracking, Model Registry Quality Gate, and Webhook Handler.
"""

import pytest
from mlops.registry.model_registry import ModelRegistryGate
from mlops.webhooks.hf_handler import HuggingFaceWebhookHandler
from mlops.tracking.mlflow_tracker import MLflowTracker


def test_model_registry_detection_promotion():
    gate = ModelRegistryGate(min_map50=0.85)

    # Superior candidate -> Approved
    champ = {"mAP@50": 0.88}
    cand = {"mAP@50": 0.92}
    approved, msg = gate.evaluate_detection_candidate(cand, champ)
    assert approved is True

    # Inferior candidate -> Rejected
    bad_cand = {"mAP@50": 0.86}
    approved, msg = gate.evaluate_detection_candidate(bad_cand, champ)
    assert approved is False


def test_model_registry_verification_promotion():
    gate = ModelRegistryGate(max_eer=0.08)

    # Lower EER -> Approved
    champ = {"eer": 0.05}
    cand = {"eer": 0.03}
    approved, msg = gate.evaluate_verification_candidate(cand, champ)
    assert approved is True

    # Higher EER -> Rejected
    bad_cand = {"eer": 0.07}
    approved, msg = gate.evaluate_verification_candidate(bad_cand, champ)
    assert approved is False


def test_webhook_handler_idempotency():
    handler = HuggingFaceWebhookHandler(secret_token="my-secret-token")
    
    assert handler.verify_secret("my-secret-token") is True
    assert handler.verify_secret("wrong-token") is False

    # First delivery: Accepted
    ok, msg = handler.process_event("event-123", {"repo": {"name": "signature-data"}})
    assert ok is True

    # Duplicate delivery: Blocked by idempotency guard
    dup_ok, dup_msg = handler.process_event("event-123", {"repo": {"name": "signature-data"}})
    assert dup_ok is False
    assert "Idempotency Guard" in dup_msg


def test_mlflow_tracker_local_fallback():
    tracker = MLflowTracker(experiment_name="test-experiment")
    run_id = tracker.log_training_run(
        run_name="test_run",
        parameters={"lr": 0.001, "batch_size": 32},
        metrics={"mAP@50": 0.91, "loss": 0.12}
    )
    assert isinstance(run_id, str)
    assert len(run_id) > 0
