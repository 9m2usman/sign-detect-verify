import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import io
from PIL import Image

from api.main import app

client = TestClient(app)

def create_test_image(size=(100, 100), color="white"):
    file = io.BytesIO()
    image = Image.new("RGB", size, color)
    image.save(file, format="JPEG")
    file.seek(0)
    return file

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "signature-detection-api"

@patch("api.main.s3_client")
@patch("api.main.sqs_client")
def test_submit_signature_verification_success(mock_sqs, mock_s3):
    mock_s3.upload_fileobj.return_value = None
    mock_sqs.send_message.return_value = {"MessageId": "mock-sqs-message-id"}

    doc_img = create_test_image()
    ref_img = create_test_image()

    response = client.post(
        "/v1/signatures/verify",
        files={
            "document": ("document.jpg", doc_img, "image/jpeg"),
            "reference_signatures": ("ref.jpg", ref_img, "image/jpeg")
        }
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "QUEUED"
    assert "job_id" in data
    assert mock_s3.upload_fileobj.call_count == 2
    assert mock_sqs.send_message.call_count == 1

def test_submit_signature_verification_invalid_mime():
    text_file = io.BytesIO(b"Not an image")
    ref_img = create_test_image()

    response = client.post(
        "/v1/signatures/verify",
        files={
            "document": ("document.txt", text_file, "text/plain"),
            "reference_signatures": ("ref.jpg", ref_img, "image/jpeg")
        }
    )

    assert response.status_code == 400
    assert "Unsupported document format" in response.json()["detail"]

@patch("api.main.s3_client")
def test_get_job_status_completed(mock_s3):
    mock_body = MagicMock()
    mock_body.read.return_value = b'{"job_id": "test-job-123", "status": "COMPLETED", "decision": "VERIFIED", "similarity_score": 0.88, "is_match": true, "detections_count": 1, "matches": []}'
    mock_s3.get_object.return_value = {"Body": mock_body}

    response = client.get("/v1/signatures/jobs/test-job-123")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-123"
    assert data["status"] == "COMPLETED"
    assert data["decision"] == "VERIFIED"
    assert data["similarity_score"] == 0.88
