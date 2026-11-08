# Research and Reference Analysis

This document provides a technical evaluation of the three core reference projects and engineering guidelines mandated in Part 4 of the project requirements:
1. **Reference 1:** `Melios22/Signature-Detect-To-Verify` (and Hugging Face model `Mels22/Signature-Detection-Verification`)
2. **Reference 2:** `SIGNUM` / `Uttpal-Tripathy/signature-verification-system`
3. **Reference 3:** Hugging Face Automated Retraining Pipeline (`webhooks-guide-auto-retrain`)

---

## Reference 1: Signature Detect To Verify (`Melios22/Signature-Detect-To-Verify`)

### 1. What the project does
The project automates end-to-end signature detection and authenticity verification from scanned documents. It accepts a raw document image, detects the region containing the signature, crops the signature, compares the crop against a pre-registered reference signature for a given user, and outputs a binary decision (`0: Genuine`, `1: Forged`).

### 2. Architecture
Two-stage sequential pipeline:
$$\text{Full Document Image} \xrightarrow{\text{Stage 1: Detection}} \text{Bounding Box Crop} \xrightarrow{\text{Stage 2: Verification}} \text{Similarity Score / Decision}$$
The detection stage localizes regions of interest, while the verification stage operates exclusively on the cropped signature images.

### 3. Models
- **Detection Model:** Fine-tuned `YOLO11s` (Ultralytics architecture).
- **Verification Model:** Custom 4-layer Convolutional Siamese Neural Network (Siamese CNN) with contrastive loss, trained to project paired signature images into Euclidean distance space.

### 4. Dataset
- **Detection:** Trained on *SignverOD* and *Mels22/SigDetectVerifyFlow* (Roboflow exports with standard YOLO bounding box labels).
- **Verification:** Trained on signature verification pairs (genuine and forged pairs).

### 5. Training Process
- Performed inside Google Colab notebooks using Ultralytics CLI for YOLO11s.
- Siamese CNN trained in PyTorch using binary cross-entropy or contrastive loss over paired batches.
- Models exported as `.pt` weights.

### 6. Evaluation Approach
- **Detection:** Standard YOLO mAP@50 and mAP@50:95.
- **Verification:** Binary accuracy, loss curves, and Euclidean distance separation between genuine and forged pairs.

### 7. Backend
Minimal script or Gradio demo; lacks a production-grade asynchronous API, database persistence, queue workers, or decoupled infrastructure.

### 8. Deployment Approach
Direct execution via Python CLI / notebook or Hugging Face Space demo. Not built for enterprise distributed scale.

### 9. Strengths
- Clear, intuitive two-stage concept (detection followed by verification).
- Lightweight signature cropping utility.
- End-to-end runnable demonstration on Hugging Face.

### 10. Weaknesses
- **Licensing Violation:** Built on Ultralytics (`YOLO11s`), which is licensed under **AGPL-3.0**. AGPL-3.0 is strictly prohibited for commercial and production deployment by our requirements.
- **Synchronous Bottleneck:** No message queue, asynchronous ingestion, or worker separation.
- **No Production MLOps:** Lacks automated retraining webhooks, experiment tracking (MLflow), or model registries.
- **Crude Thresholding:** Uses hardcoded distance cutoffs without validation calibration or support for a "REVIEW" uncertainty tier.

### 11. What we can learn
- The two-stage paradigm (detection cropping $\rightarrow$ verification) is sound and optimal.
- Contrastive learning with Siamese architectures effectively distinguishes stroke nuances when paired appropriately.

### 12. What we will improve
1. **Strictly Permissive Models:** Replace AGPL Ultralytics with **RTMDet-Tiny** (Apache 2.0 via OpenMMLab) and **YOLOX-Tiny** (Apache 2.0).
2. **Advanced Metric Learning:** Implement both Siamese CNN (contrastive) and Triplet Networks (anchor, positive, negative with Triplet Margin Loss) and evaluate both.
3. **Calibrated Thresholds:** Support three-tier decisions (`GENUINE`, `FORGED`, `REVIEW`) with empirically calibrated thresholds on validation sets.
4. **Multiple References:** Support multiple signature instances per document and multiple reference comparisons with calibrated aggregation.
5. **Enterprise Architecture:** Decoupled FastAPI backend, Celery workers, RabbitMQ broker, and PostgreSQL persistence.

---

## Reference 2: SIGNUM Signature Verification System

### 1. What the project does
SIGNUM provides a structured backend service for signature verification, accepting document submissions, managing identity references, comparing signatures, and serving verification results over an API.

### 2. Architecture
Client-Server REST architecture built around FastAPI, exposing endpoints for document upload, job submission, and result retrieval.

### 3. Models
Embedding-based signature verification backbones (ResNet or custom CNN feature extractors) projecting signatures into normalized vector spaces, followed by cosine similarity.

### 4. Dataset
Trained on offline signature benchmarks (such as CEDAR or BHSig) with author identity labels.

### 5. Training Process
Offline supervised metric learning, checkpointing PyTorch models, and serving weights via backend services.

### 6. Evaluation Approach
Evaluation using accuracy, precision, recall, and false acceptance/rejection metrics on validation splits.

### 7. Backend
FastAPI with modular routers, Pydantic schemas for request validation, structured logging, and test fixtures.

### 8. Deployment Approach
Dockerized containerized application deployable on cloud or local Docker environments.

### 9. Strengths
- Clean FastAPI routing and modular project structure.
- Clear separation of schemas, routers, and model inference services.
- Solid automated testing patterns for HTTP status codes and payloads.

### 10. Weaknesses
- Often executes model inference synchronously within request handlers, risking API timeouts under heavy load.
- Lacks a distributed message queue (RabbitMQ/Celery) for decoupled processing.
- Lacks an integrated object detection stage to isolate signatures from full multi-page documents.
- Lacks continuous training triggers or MLOps lifecycle tracking.

### 11. What we can learn
- Clean FastAPI application structure (`api/`, `core/`, `schemas/`, `services/`, `models/`).
- Standardized error handling, HTTP status codes, and Pydantic validation schemas.

### 12. What we will improve
1. **Asynchronous Decoupling:** Move all deep learning inference out of the FastAPI process into a dedicated **Celery worker** communicating over **RabbitMQ**.
2. **Unified Pipeline:** Combine object detection with verification so users submit whole documents without needing pre-cropped signatures.
3. **Persistent Job Store:** Store job states, execution metadata, and bounding boxes in **PostgreSQL** with SQLAlchemy.
4. **Resilient Queues:** Implement message acknowledgments, task retries, dead-letter exchanges, and worker heartbeat monitoring.

---

## Reference 3: Hugging Face Automatic Retraining (`webhooks-guide-auto-retrain`)

### 1. What the project does
The Hugging Face guide demonstrates an event-driven MLOps retraining architecture. When a dataset repository on the Hugging Face Hub receives new commits or updated data splits, a webhook event is dispatched to trigger automated retraining of downstream models.

### 2. Architecture
$$\text{HF Dataset Repo Update} \xrightarrow{\text{Webhook POST}} \text{FastAPI Endpoint} \xrightarrow{\text{Auth \& Idempotency}} \text{Remote Compute Job} \xrightarrow{\text{Model Registry \& Deployment}}$$

### 3. Models
Demonstrated with image classification (e.g., `microsoft/resnet-50`) through Hugging Face AutoTrain or custom container training jobs.

### 4. Dataset
Hugging Face Hub hosted datasets with revision commit hashes (`refs/convert/parquet` or Git commit SHAs).

### 5. Training Process
Event triggers an HTTP POST request containing repository metadata, event type, and commit SHA. The webhook receiver verifies the secret, validates the payload, and dispatches an asynchronous training job.

### 6. Evaluation Approach
Automated post-training validation against a held-out test split, comparing candidate metrics against the production baseline.

### 7. Backend
FastAPI server running in Docker, exposing a `/webhook` endpoint with HMAC-SHA256 signature verification or `X-Webhook-Secret` header validation.

### 8. Deployment Approach
Hosted on Hugging Face Spaces or containerized cloud compute (AWS Batch, ECS, or EC2 GPU).

### 9. Strengths
- Zero human intervention required to kick off retraining when datasets expand.
- Clear, auditable lineage linking dataset commit SHA to trained model artifacts.
- Cryptographic verification via webhook secrets prevents unauthorized execution.

### 10. Weaknesses
- The reference tutorial assumes simple auto-train containers without comprehensive model validation gates.
- Missing rollback logic if a newly trained model underperforms the existing production model.
- Does not distinguish between detection dataset updates vs. verification dataset updates.

### 11. What we can learn
- Secure webhook ingestion requires constant-time secret validation (`hmac.compare_digest`).
- Webhook endpoints must be idempotent: duplicate webhook delivery (common in distributed systems) must not launch duplicate redundant training jobs.

### 12. What we will improve
1. **Dual Pipeline Routing:** Distinguish between detection dataset updates (triggering RTMDet/YOLOX retraining) and verification dataset updates (triggering Siamese/Triplet retraining).
2. **Quality Gates & Model Registry:** Candidate models must strictly exceed the production champion's mAP@50 or EER on benchmark sets before promotion.
3. **Automated Rollback:** If a deployed model fails health checks or verification latency tests, the deployment automatically rolls back to the prior stable version.
4. **MLflow Tracking:** All parameters, dataset SHAs, GPU run times, loss curves, and artifact checkpoints logged to an MLflow tracking server.
