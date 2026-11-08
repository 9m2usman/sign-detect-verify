# Manual Testing & Verification Guide

## 1. Local Environment Pre-flight Check

Ensure local dependencies and containers are functioning:
```bash
# Verify Python version (3.11 - 3.13)
python --version

# Verify Docker containers are up
docker-compose ps
```

## 2. API Verification Steps

### Step 1: Check System Health
```bash
curl -X GET http://localhost:8000/health
```
Expected output: `{"status":"healthy","service":"signature-backend","version":"1.0.0"}`

### Step 2: Test Document Signature Detection
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@tests/sample_doc.png"
```
Verify that a `202 Accepted` response is returned with a `job_id`.

### Step 3: Test Pairwise Signature Verification
```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "query_image=@tests/query_sign.png" \
  -F "reference_image=@tests/ref_sign.png"
```
Verify that a `202 Accepted` response is returned with a `job_id`.

### Step 4: Verify Job Results
```bash
curl -X GET http://localhost:8000/api/v1/jobs/<job_id>
```
Verify status changes from `QUEUED` / `PROCESSING` to `COMPLETED` with similarity metrics.
