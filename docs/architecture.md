# End-to-End System Architecture

## 1. System Components

The Signature Detection & Verification System is organized into three decoupled layers:

### 1.1 Ingestion & API Layer (FastAPI)
- **Role**: Validates HTTP multipart requests, handles document uploads, manages writer registrations, and dispatches jobs to the queue.
- **Persistence**: Relational metadata, writers, signatures, and job state stored in PostgreSQL.

### 1.2 Queue & Worker Tier (Celery + RabbitMQ)
- **Role**: Decouples long-running ML inference from the synchronous HTTP cycle.
- **Tasks**:
  - `detect_signatures`: Loads document, runs Faster R-CNN, extracts crops, saves bounding boxes.
  - `verify_signatures`: Compares cropped signature embedding against enrolled reference signatures, computes cosine similarity and verification decision.

### 1.3 MLOps & Experiment Tracking (MLflow)
- **Role**: Centralized parameter tracking, metric logging (mAP, EER, loss), and model checkpoint versioning.

## 2. Asynchronous Flow Diagram

```
User Document Upload
       │
       ▼
 [FastAPI API]  ───▶ [PostgreSQL] (Job status = QUEUED)
       │
       ▼ (Publish message)
  [RabbitMQ]
       │
       ▼ (Consume task)
[Celery Worker]
       ├─▶ Step 1: Preprocessing & Faster R-CNN Detection
       ├─▶ Step 2: Signature Crop Extraction & Normalization
       ├─▶ Step 3: Siamese ResNet Verification & Cosine Score
       └─▶ Step 4: Write Results to PostgreSQL (Job status = COMPLETED)
```
