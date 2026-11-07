import os
import uuid
import json
import time
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from fastapi.responses import JSONResponse
from api.schemas import VerificationJobSubmissionResponse, JobStatusResponse

app = FastAPI(
    title="Signature Detection & Verification API",
    description="Production-grade asynchronous signature detection and verification engine.",
    version="1.0.0"
)

# Configuration from Environment
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # For LocalStack
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "signature-verification-documents")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "http://localhost:4566/000000000000/signature-job-queue")

# Boto3 client initialization with support for LocalStack and AWS Cloud
boto_config = {"region_name": AWS_REGION}
if AWS_ENDPOINT_URL:
    boto_config["endpoint_url"] = AWS_ENDPOINT_URL

s3_client = boto3.client("s3", **boto_config)
sqs_client = boto3.client("sqs", **boto_config)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "service": "signature-detection-api",
        "timestamp": time.time()
    }

@app.post(
    "/v1/signatures/verify",
    response_model=VerificationJobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Verification"]
)
async def submit_signature_verification(
    document: UploadFile = File(..., description="Full document image (PNG/JPEG)"),
    reference_signatures: List[UploadFile] = File(..., description="One or more authentic reference signature images")
):
    # 1. Validate Mime Types
    if document.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported document format '{document.content_type}'. Must be one of: {ALLOWED_MIME_TYPES}"
        )

    for ref in reference_signatures:
        if ref.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported reference format '{ref.content_type}'. Must be one of: {ALLOWED_MIME_TYPES}"
            )

    job_id = str(uuid.uuid4())
    doc_key = f"raw-documents/{job_id}/{document.filename}"
    ref_keys = []

    try:
        # 2. Upload Document & References directly to S3 (Keep binaries out of SQS!)
        s3_client.upload_fileobj(document.file, S3_BUCKET, doc_key)

        for ref in reference_signatures:
            ref_k = f"references/{job_id}/{ref.filename}"
            s3_client.upload_fileobj(ref.file, S3_BUCKET, ref_k)
            ref_keys.append(ref_k)

        # 3. Dispatch Job Pointer to SQS
        job_message = {
            "job_id": job_id,
            "s3_bucket": S3_BUCKET,
            "document_s3_key": doc_key,
            "reference_s3_keys": ref_keys,
            "submitted_at": time.time()
        }

        sqs_client.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(job_message)
        )

        return VerificationJobSubmissionResponse(
            job_id=job_id,
            status="QUEUED",
            message="Images staged successfully. Task dispatched to SQS queue.",
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"AWS storage/queue error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected submission error: {str(e)}")


@app.get("/v1/signatures/jobs/{job_id}", response_model=JobStatusResponse, tags=["Verification"])
async def get_job_status(job_id: str):
    result_key = f"results/{job_id}.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=result_key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        return JobStatusResponse(**data)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "NoSuchKey":
            return JobStatusResponse(
                job_id=job_id,
                status="PROCESSING",
                decision="PENDING",
                error_message=None
            )
        raise HTTPException(status_code=500, detail=f"S3 query error: {str(e)}")
