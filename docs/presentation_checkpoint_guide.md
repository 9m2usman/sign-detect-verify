# Checkpoint Presentation Guide — Group 4
## Signature Detection & Verification Platform
**Batch Group:** Group 4 (5 Members: Usman [Lead], Junaid, Saim, Ayesha, Sobia)  
**Checkpoint Time:** Today 11:30 PM  
**Interactive Jira Board:** [`docs/jira_board.html`](file:///E:/sign_detect-verify/docs/jira_board.html)

---

## 1. How to Open & Present Your Jira Board
Open the interactive HTML board directly in any browser:
```powershell
# Open the interactive board in your default browser:
Start-Process "E:\sign_detect-verify\docs\jira_board.html"
```
**Key Highlights on the Board:**
- **12 Epics & 24 User Stories** (112 Story Points total).
- Clear 4-day milestone roadmap:
  - **Saturday 11:30 PM (Today):** Inspection, Dataset Separation, ML Architecture & Jira Board.
  - **Sunday:** ML Training, Model Verification, and Backend Ingestion.
  - **Monday (Holiday):** Automated MLOps Retraining Pipeline & Hugging Face Webhooks.
  - **Tuesday:** Final End-to-End Deployed Presentation & Performance Benchmarking.
- Every story has business acceptance criteria, technical tasks, assigned owners, and story point sizing.

---

## 2. Key Talking Points for Instructor Checkpoint Questions

### Q1: "Which object detection model are you using, and what about the license?"
> *"We standardise on **Faster R-CNN with a ResNet-50-FPN backbone** under the **BSD-3-Clause** license, and evaluated **RTMDet-Tiny (Apache-2.0)**. We strictly reject Ultralytics YOLO because its AGPL-3.0 license prohibits proprietary production deployment without open-sourcing the entire proprietary codebase. Our system is 100% permissively licensed."*

### Q2: "How are you handling the two datasets?"
> *"We maintain strict dataset separation:
> - **Dataset A (`netsol-signature-detection/Signature-Detection-System`):** Contains 28,958 document images with 38,094 bounding boxes and 1,285 hard-negative backgrounds. It contains **zero author IDs or genuine/forged labels**, so it is exclusively used for object detection.
> - **Dataset B (CEDAR Benchmark):** Contains 55 writers with 24 genuine and 24 skilled forgeries each (2,640 signatures). It is used exclusively for training and evaluating our deep metric Siamese verification model."*

### Q3: "How does the verification model work? Does it use pixel IoU?"
> *"No, pixel IoU or SSIM fails on handwriting because human signatures have natural translation and stroke variations (resulting in low raw IoU of ~0.04 even on genuine signatures). Instead, we use a **Siamese ResNet-50 with contrastive loss**. It projects cropped signatures onto a 256-dimensional unit hypersphere and computes translation-invariant cosine similarity with calibrated decision thresholds."*

### Q4: "How does the automated retraining pipeline work when images are uploaded to Hugging Face?"
> *"We built an automated MLOps pipeline:
> 1. Hugging Face dataset updates send an HTTP push event to our FastAPI endpoint (`/api/v1/webhooks/hf/huggingface`).
> 2. The endpoint verifies cryptographic HMAC tokens (`X-Webhook-Secret`) and enforces idempotency to prevent duplicate runs.
> 3. The `RetrainingOrchestrator` triggers a remote GPU retraining job.
> 4. Our **Model Registry Quality Gate** compares the new candidate against the active champion (candidate mAP@50 must exceed 0.8800).
> 5. If approved, the artifact is saved outside Git with a SHA-256 manifest and promoted in our database and MLflow experiment registry."*

### Q5: "What is your deployment and queue architecture?"
> *"We decouple HTTP ingestion from heavy neural computation using an asynchronous worker queue:
> - **FastAPI REST API:** Runs in ECS Fargate behind an Application Load Balancer. It receives uploads, generates a job UUID, and immediately returns HTTP 202 Accepted.
> - **RabbitMQ & Celery Workers:** Background worker instances pull verification jobs, execute Faster R-CNN detection, crop signature candidates, compute Siamese similarity scores, and persist results to **PostgreSQL**.
> - **Observability:** Prometheus scrapes real-time metrics (`/metrics`) and feeds our Grafana dashboard with request latency (p50, p95, p99), throughput, and genuine-to-forged ratios."*

---

## 3. Demo Commands Ready to Run Live If Asked

### Live Demo 1: Full Pipeline CLI
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" "E:\sign_detect-verify\verify_document.py" --doc "E:\sign_detect-verify\sample_test_run\sample_document.jpg" --ref "E:\sign_detect-verify\sample_test_run\sample_document.jpg" --crops-dir "C:\Users\Itcomplex\AppData\Local\Temp\extracted_crops"
```

### Live Demo 2: All 32 Tests Passing
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -m pytest "E:\sign_detect-verify\tests\" -o cache_dir="C:\Users\Itcomplex\AppData\Local\Temp\pytest_cache"
```

### Live Demo 3: Prometheus Observability Telemetry
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -u "C:\Users\Itcomplex\.gemini\antigravity\brain\a83f09a5-a162-400a-9fbc-443f4a7bb828\scratch\test_phase6_observability.py"
```
