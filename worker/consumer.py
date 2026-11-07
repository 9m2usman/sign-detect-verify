import os
import json
import time
import io
import boto3
from botocore.exceptions import ClientError
from PIL import Image

from ml.engine import SignatureEngine
from ml.detector import RTDETRSignatureDetector
from ml.verification import ResNet50SignatureVerifier

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "signature-verification-documents")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "http://localhost:4566/000000000000/signature-job-queue")

boto_config = {"region_name": AWS_REGION}
if AWS_ENDPOINT_URL:
    boto_config["endpoint_url"] = AWS_ENDPOINT_URL

s3_client = boto3.client("s3", **boto_config)
sqs_client = boto3.client("sqs", **boto_config)

def download_s3_image(bucket: str, key: str) -> Image.Image:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

def run_worker_loop():
    print(f"[*] Starting Signature Verification Worker...")
    print(f"[*] SQS Queue: {SQS_QUEUE_URL}")
    print(f"[*] S3 Bucket: {S3_BUCKET}")

    # Initialize Engine (RT-DETR + ResNet-50)
    # Allows hot-swapping or version updates
    print("[*] Loading RT-DETR Detection & ResNet50 Feature Extractor...")
    detector = RTDETRSignatureDetector()
    verifier = ResNet50SignatureVerifier()
    engine = SignatureEngine(detector=detector, verifier=verifier, verification_threshold=0.82)
    print("[+] Models initialized. Listening for messages...")

    while True:
        try:
            # Long-poll SQS
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=120
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for msg in messages:
                receipt_handle = msg["ReceiptHandle"]
                body = json.loads(msg["Body"])
                job_id = body["job_id"]
                bucket = body["s3_bucket"]
                doc_key = body["document_s3_key"]
                ref_keys = body["reference_s3_keys"]

                print(f"[+] Processing Job: {job_id}")

                try:
                    # 1. Fetch Images from S3
                    doc_img = download_s3_image(bucket, doc_key)
                    ref_imgs = [download_s3_image(bucket, rk) for rk in ref_keys]

                    # 2. Execute Detection & Verification Pipeline
                    result = engine.process_document(doc_img, ref_imgs)
                    result["job_id"] = job_id
                    result["status"] = "COMPLETED"

                    # 3. Persist Result to S3
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=f"results/{job_id}.json",
                        Body=json.dumps(result, indent=2),
                        ContentType="application/json"
                    )

                    # 4. Delete processed message
                    sqs_client.delete_message(
                        QueueUrl=SQS_QUEUE_URL,
                        ReceiptHandle=receipt_handle
                    )
                    print(f"[✓] Job {job_id} successfully completed. Decision: {result['decision']}")

                except Exception as proc_err:
                    print(f"[!] Processing failed for job {job_id}: {proc_err}")
                    # Save failure status to S3
                    fail_record = {
                        "job_id": job_id,
                        "status": "FAILED",
                        "error_message": str(proc_err)
                    }
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=f"results/{job_id}.json",
                        Body=json.dumps(fail_record),
                        ContentType="application/json"
                    )
                    # SQS will retry up to DLQ maxReceiveCount

        except Exception as queue_err:
            print(f"[!] SQS polling error: {queue_err}")
            time.sleep(5)

if __name__ == "__main__":
    run_worker_loop()
