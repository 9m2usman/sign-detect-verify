# API Reference Documentation

**Service:** Signature Detection, Verification & Automated MLOps Backend  
**Base URL:** `http://localhost:8000/api/v1`  
**Interactive Swagger UI:** `http://localhost:8000/docs`  
**Interactive ReDoc:** `http://localhost:8000/redoc`

---

## 1. Authentication & Headers

| Header | Description | Required Endpoints |
| :--- | :--- | :--- |
| `Content-Type` | `multipart/form-data` for file uploads, `application/json` for webhooks | All POST endpoints |
| `X-Webhook-Secret` | Shared secret token for authenticating automated retraining webhooks | `POST /webhooks/huggingface` |

---

## 2. Endpoints Reference

### 2.1 Submit Document for Signature Verification
- **Path:** `POST /api/v1/signatures/verify`
- **Method:** `POST`
- **Status Code:** `202 Accepted`
- **Payload:** `multipart/form-data`
  - `document`: Image file (JPEG/PNG) of the document to inspect.
  - `references`: One or more authentic reference signature image files.

#### Example Request (`curl`)
```bash
curl -X POST "http://localhost:8000/api/v1/signatures/verify" \
  -F "document=@document_loan.jpg" \
  -F "references=@reference_signer_1.jpg" \
  -F "references=@reference_signer_2.jpg"
```

#### Example Response (`202 Accepted`)
```json
{
  "job_id": "97a44f08-3324-48d8-bdec-f2853b262a6a",
  "status": "PENDING",
  "message": "Verification job successfully submitted and queued for processing.",
  "document_filename": "document_loan.jpg",
  "reference_filenames": [
    "reference_signer_1.jpg",
    "reference_signer_2.jpg"
  ],
  "poll_url": "/api/v1/jobs/97a44f08-3324-48d8-bdec-f2853b262a6a"
}
```

---

### 2.2 Poll Job Status & Verification Results
- **Path:** `GET /api/v1/jobs/{job_id}`
- **Method:** `GET`
- **Status Code:** `200 OK` (or `404 Not Found` if job ID does not exist)

#### Example Request (`curl`)
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/97a44f08-3324-48d8-bdec-f2853b262a6a"
```

#### Example Response (`200 OK - Processing Completed`)
```json
{
  "job_id": "97a44f08-3324-48d8-bdec-f2853b262a6a",
  "status": "COMPLETED",
  "detected_count": 1,
  "created_at": "2026-11-08T18:50:00Z",
  "completed_at": "2026-11-08T18:50:02Z",
  "results": [
    {
      "signature_index": 0,
      "bbox": [352.0, 480.0, 576.0, 563.2],
      "confidence_detection": 0.96,
      "reference_filename": "uploads/ref_0_reference_signer_1.jpg",
      "similarity_score": 0.9421,
      "decision": "GENUINE",
      "crop_url": "uploads/crops_97a44f08/crop_sig_0.jpg"
    }
  ],
  "error_message": null
}
```

---

### 2.3 Active Production Model Information
- **Path:** `GET /api/v1/models/info`
- **Method:** `GET`
- **Status Code:** `200 OK`

#### Example Request (`curl`)
```bash
curl -X GET "http://localhost:8000/api/v1/models/info"
```

#### Example Response
```json
{
  "detector_model": "RTMDet-Tiny (OpenMMLab)",
  "detector_license": "Apache License 2.0",
  "detector_parameters": "4.8 Million Parameters (CSPNeXt Backbone)",
  "verifier_model": "Siamese CNN with Contrastive Metric Projection (ResNet-18)",
  "verifier_license": "BSD-3-Clause / Apache 2.0",
  "verifier_parameters": "11.2 Million Parameters (256-dimensional Unit Hypersphere)",
  "threshold_genuine": 0.82,
  "threshold_forged": 0.65,
  "compliance_notice": "100% Permissive Open Source (Strictly Non-Ultralytics, Zero AGPL)."
}
```

---

### 2.4 Hugging Face Webhook Ingestion
- **Path:** `POST /api/v1/webhooks/huggingface`
- **Method:** `POST`
- **Status Code:** `200 OK` (`401 Unauthorized` if secret invalid)

#### Example Request (`curl`)
```bash
curl -X POST "http://localhost:8000/api/v1/webhooks/huggingface" \
  -H "X-Webhook-Secret: dev-webhook-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "event": "update",
    "repo": {
      "name": "org/signature-detection-dataset",
      "type": "dataset"
    }
  }'
```

#### Example Response
```json
{
  "status": "triggered",
  "retraining_target": "detection_rtmdet",
  "repo": "org/signature-detection-dataset",
  "message": "Automated remote GPU retraining job queued for target: detection_rtmdet."
}
```
