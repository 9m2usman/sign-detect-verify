# Production AWS Architecture & Infrastructure Blueprint

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Target Environment:** AWS Cloud (Production) & LocalStack (Local Dev)  
**Standard:** Well-Architected Framework (Reliability, Security, Performance Efficiency, Cost Optimization)

---

## 1. System Architecture Overview

The system is architected as an asynchronous, event-driven, decoupled microservices topology:
1. **Inference Ingestion Layer:** A high-throughput, stateless FastAPI tier running on AWS ECS Fargate behind an Application Load Balancer (ALB).
2. **Persistence & Decoupling Layer:** Direct binary storage in Amazon S3 combined with lightweight job orchestration via Amazon SQS.
3. **ML Inference Worker Pool:** Horizontally scalable GPU or CPU workers on AWS ECS/EC2 executing detection (YOLOX-S / RT-DETR) and verification (ResNet50 / ViT-B/16).
4. **Automated Continuous Training (MLOps) Layer:** Event-driven pipeline triggered by Hugging Face Dataset webhooks, running remote training jobs with Comet ML / MLflow tracking, promoting vetted weights to an S3 Model Registry, and triggering rolling inference reloads.

```
                              [ HTTPS Client / Web UI ]
                                          |
                                          v
                              [ AWS Route 53 / CDN ]
                                          |
                                          v
                         [ Application Load Balancer (ALB) ]
                                          |
                                          v
                      [ FastAPI Service (AWS ECS Fargate) ]
                             /                         \
           1. Upload Raw    /                           \ 2. Push Metadata Pointer
              Images       /                             \   {job_id, s3_keys}
                          v                               v
             +-----------------------+         +-----------------------+
             |   Amazon S3 Bucket    |         |   Amazon SQS Queue    |
             |   - /raw-documents/   |         |   (Vis. Timeout: 120s)|
             |   - /crops/           |         +-----------------------+
             |   - /models/          |                     |
             +-----------------------+                     | 3. Long Polling (20s)
                       ^              ^                    v
                       |              |        +-----------------------+
                       |              +------- | Worker Service (ECS)  |
                       |       Fetch Images    | - YOLOX-S / RT-DETR   |
                       |       & Write Results | - ResNet50 / ViT Embed|
                       |                       +-----------------------+
                       |                                   |
                       |                                   | 4. Update Status
                       |                                   v
                       +---------------------- [ DynamoDB / S3 Results ]

 ====================================================================================
                       AUTOMATED MLOps TRAINING & RELOAD LOOP
 ====================================================================================

    +--------------------------+
    |  Hugging Face Dataset    |
    |  (New images committed)  |
    +--------------------------+
                 |
                 | Webhook (commit / push event)
                 v
    +--------------------------+
    |  GitHub Actions Runner   |
    +--------------------------+
                 |
                 | Dispatches Remote GPU Job
                 v
    +--------------------------------------------------------+
    |   Remote Training Job (AWS SageMaker / Colab / EC2)    |
    +--------------------------------------------------------+
              /                                    \
             / Logs Epoch Metrics                   \ Saves Best Weights
            v                                        v
  +--------------------+                   +------------------------------+
  |  Comet ML / MLflow |                   | S3 Model Registry            |
  |  (mAP, Loss, EER)  |                   | (s3://sig-model-checkpoints) |
  +--------------------+                   +------------------------------+
                                                          |
                                                          | Zero-downtime Rollout
                                                          v
                                           +------------------------------+
                                           | Inference Worker Reload      |
                                           +------------------------------+
```

---

## 2. Component Specifications

### 2.1 Amazon S3 (Object Storage)
* **Bucket Layout:**
  - `s3://sig-verification-prod/raw-documents/{job_id}/`: Uploaded document images.
  - `s3://sig-verification-prod/references/{job_id}/`: Known reference signature images.
  - `s3://sig-verification-prod/cropped-signatures/{job_id}/`: Cropped signature images extracted by the detector.
  - `s3://sig-verification-prod/results/{job_id}.json`: Final evaluation results.
  - `s3://sig-model-registry/models/{architecture}/{version}/`: Production model checkpoints.
* **Lifecycle Policy:** Raw documents transition to S3 Standard-Infrequent Access after 30 days, expire after 90 days.

### 2.2 Amazon SQS (Decoupled Message Queue)
* **Design Rule:** **Never pass binary images through SQS.** SQS payloads are strictly under 2 KB, containing only metadata pointers:
  ```json
  {
    "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    "bucket": "sig-verification-prod",
    "document_key": "raw-documents/f81d4fae/doc.jpg",
    "reference_key": "references/f81d4fae/ref.jpg",
    "timestamp": 1731000000
  }
  ```
* **Queue Configuration:**
  - `DefaultVisibilityTimeout`: **120 seconds** (allows worker sufficient time for inference and S3 writing).
  - `ReceiveMessageWaitTimeSeconds`: **20 seconds** (long-polling to minimize empty receive API costs).
  - `RedrivePolicy`: Routes to Dead Letter Queue (`sig-job-dlq`) after **3 consecutive processing failures**.

### 2.3 Worker Pool & Zero-Downtime Model Reloading
* Workers run on an event loop listening to SQS.
* Workers query the S3 Model Registry on container startup or upon receiving an SQS model reload broadcast event.
* In-flight jobs are completed on the current model before memory weights are hot-swapped, guaranteeing zero dropped tasks.
