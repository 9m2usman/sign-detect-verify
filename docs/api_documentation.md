# API Documentation & Endpoint Specification

## 1. General Principles
- Base URL: `/api/v1`
- Content Types: `multipart/form-data` for file uploads, `application/json` for payloads and responses.
- Authentication: Bearer JWT token header (`Authorization: Bearer <token>`).

## 2. Endpoints

### 2.1 Health Check
- **GET** `/health`
- **Response** `200 OK`:
  ```json
  {
    "status": "healthy",
    "service": "signature-backend",
    "version": "1.0.0"
  }
  ```

### 2.2 Detect Signatures
- **POST** `/api/v1/detect`
- **Request**: Multipart form with `file` (document image/PDF).
- **Response** `202 Accepted`:
  ```json
  {
    "job_id": "c1f7a28e-8914-41d3-a5c9-2ef531201948",
    "status": "QUEUED",
    "message": "Signature detection task submitted successfully."
  }
  ```

### 2.3 Verify Signature Pair
- **POST** `/api/v1/verify`
- **Request**: Multipart form with `query_image` and `reference_image` (or `writer_id`).
- **Response** `202 Accepted`:
  ```json
  {
    "job_id": "e8d64132-72bc-44fa-9ec8-6cb493721098",
    "status": "QUEUED",
    "message": "Signature verification task submitted successfully."
  }
  ```

### 2.4 Query Job Status
- **GET** `/api/v1/jobs/{job_id}`
- **Response** `200 OK`:
  ```json
  {
    "job_id": "e8d64132-72bc-44fa-9ec8-6cb493721098",
    "status": "COMPLETED",
    "result": {
      "similarity_score": 0.884,
      "verdict": "GENUINE",
      "threshold": 0.75,
      "confidence": 0.94
    }
  }
  ```
