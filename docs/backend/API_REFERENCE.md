# FastAPI Backend & REST API Reference

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Standard:** OpenAPI 3.0 / RESTful Design  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)  

---

## 1. API Architecture & Design Rules

1. **Strictly Non-Blocking:** High-resolution document images are uploaded directly to object storage (Amazon S3). SQS messages contain only small metadata references ($< 2\text{ KB}$).
2. **HTTP 202 Accepted:** Verification tasks return `202 Accepted` immediately with a `job_id` to prevent client timeouts during complex image inference.
3. **MIME Type Validation:** Rejects unsupported non-image formats (`text/plain`, executables) before staging.

---

## 2. Endpoints Reference

### `GET /health`
* **Summary:** Health check endpoint for AWS Application Load Balancer (ALB) target groups.
* **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "signature-detection-api",
    "timestamp": 1731000000.123
  }
  ```

---

### `POST /v1/signatures/verify`
* **Summary:** Submit document image and one or more reference signatures for asynchronous verification.
* **Content-Type:** `multipart/form-data`
* **Parameters:**
  * `document` (File, Required): Scanned document containing candidate signatures (JPEG, PNG).
  * `reference_signatures` (List[File], Required): One or more authentic reference signatures from the registered signer.
* **Response (202 Accepted):**
  ```json
  {
    "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    "status": "QUEUED",
    "message": "Images staged successfully. Task dispatched to SQS queue.",
    "submitted_at": "2026-11-07T12:00:00Z"
  }
  ```

---

### `GET /v1/signatures/jobs/{job_id}`
* **Summary:** Query processing status and final results of a verification job.
* **Response (200 OK - Processing):**
  ```json
  {
    "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    "status": "PROCESSING",
    "decision": "PENDING",
    "error_message": null
  }
  ```
* **Response (200 OK - Completed):**
  ```json
  {
    "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    "status": "COMPLETED",
    "decision": "VERIFIED",
    "similarity_score": 0.8942,
    "is_match": true,
    "detections_count": 1,
    "matches": [
      {
        "candidate_index": 0,
        "box_xyxy": [340, 520, 610, 680],
        "detection_confidence": 0.9421,
        "similarity_score": 0.8942,
        "matched_reference_index": 0,
        "is_match": true,
        "decision": "GENUINE_MATCH"
      }
    ],
    "processing_time_sec": 0.0842
  }
  ```
