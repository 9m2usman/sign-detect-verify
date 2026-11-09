# Jira Project Management & Team Work Breakdown Structure

**Project:** Signature Detection & Verification Platform  
**Team Members & Roles:**
- **Usman (Project Lead & Cloud Architect):** System architecture, AWS deployment, MLOps orchestration, security.
- **Junaid (ML Engineer - Detection):** Dataset A audit, YOLO conversion, Faster R-CNN / RTMDet GPU training.
- **Saim (MLOps & Evaluation):** Alternative detection models, MLflow tracking, mAP evaluation, model registry quality gates.
- **Ayesha (ML Engineer - Verification):** Verification dataset research (CEDAR/BHSig), Siamese ResNet metric learning, EER/ROC-AUC evaluation, SSIM baseline.
- **Sobia (Backend & Data Engineer):** FastAPI endpoints, PostgreSQL schema/migrations, RabbitMQ broker, Celery worker.

---

## High-Level Epics & Story Breakdown

### Epic 1: Project Planning, Standards & Architecture Baseline
- **Epic Owner:** Usman
- **Story 1.1: Licensing & Dependency Audit** (Owner: Usman | Points: 3)
  - *Description:* Verify permissive licenses (MIT, Apache 2.0, BSD-3-Clause) for all frameworks, models, and dependencies. Enforce zero AGPL / Ultralytics code.
  - *Acceptance Criteria:* `LICENSES.md` updated and validated against all imports in `requirements.txt`.
- **Story 1.2: API & Schema Standardization** (Owner: Sobia | Points: 3)
  - *Description:* Define standard Pydantic v2 schemas for document upload, verification, and job polling.
  - *Acceptance Criteria:* Schemas pass validation; Swagger docs display clear request/response models.

---

### Epic 2: Dataset Research, Audit & Acquisition
- **Epic Owner:** Junaid & Ayesha
- **Story 2.1: Dataset A Annotation Audit & Bug Fix** (Owner: Junaid | Points: 5)
  - *Description:* Fix the critical defect in `training/detection/train_rtdetr.py` where blank images receive full-image bounding boxes. Verify 28,958 images across train/val/test splits.
  - *Acceptance Criteria:* Unannotated images produce empty target tensors (`boxes: [0, 4]`, `labels: [0]`).
- **Story 2.2: Dataset B Research & Candidate Comparison** (Owner: Ayesha | Points: 5)
  - *Description:* Conduct rigorous comparative evaluation of CEDAR vs. BHSig260 (signer count, genuine/forged balance, format, commercial terms, writer-independent split).
  - *Acceptance Criteria:* Present written recommendation for user approval before downloading any verification dataset.

---

### Epic 3: Signature Detection Model & Training
- **Epic Owner:** Junaid & Saim
- **Story 3.1: Reproducible GPU Training Notebook** (Owner: Junaid | Points: 8)
  - *Description:* Develop modular, reproducible Colab/GPU training notebook for Faster R-CNN / RTMDet-Tiny with MLflow tracking.
  - *Acceptance Criteria:* Notebook runs end-to-end, logs loss curves and metrics to MLflow, and exports checkpoint.
- **Story 3.2: Detection Evaluation & Weight Integration** (Owner: Saim | Points: 5)
  - *Description:* Evaluate trained detector on test split (Precision, Recall, mAP@0.5, mAP@0.5:0.95, latency). Integrate checkpoint into inference pipeline.
  - *Acceptance Criteria:* Neural detection scores exceed 0.80 on test signatures; heuristic fallback is completely eliminated for valid signatures.

---

### Epic 4: Signature Verification Metric Model
- **Epic Owner:** Ayesha
- **Story 4.1: Real Siamese ResNet Metric Learning** (Owner: Ayesha | Points: 8)
  - *Description:* Build and train Siamese ResNet-50 feature extractor with contrastive/triplet loss on real signature pairs (replacing the mock `torch.rand()` script).
  - *Acceptance Criteria:* Generates 256-dimensional L2-normalized embeddings; loss decreases monotonically on real verification pairs.
- **Story 4.2: SSIM Baseline & Verification Evaluation** (Owner: Ayesha | Points: 5)
  - *Description:* Benchmark Siamese cosine similarity against traditional SSIM baseline. Measure EER, FAR, FRR, ROC-AUC on validation split and calibrate threshold.
  - *Acceptance Criteria:* EER and ROC curve documented in MLflow; decision threshold calibrated on validation data.

---

### Epic 5: Backend REST API & Relational Persistence
- **Epic Owner:** Sobia
- **Story 5.1: FastAPI Asynchronous Microservice** (Owner: Sobia | Points: 5)
  - *Description:* Implement REST API endpoints (`POST /documents/upload`, `POST /signatures/verify`, `GET /jobs/{id}`, `GET /health`).
  - *Acceptance Criteria:* Endpoints return standardized JSON responses with unique job IDs and proper HTTP status codes.
- **Story 5.2: PostgreSQL Schema & Database Migrations** (Owner: Sobia | Points: 5)
  - *Description:* Configure SQLAlchemy models and Alembic migrations for jobs, detections, verification results, and model versions.
  - *Acceptance Criteria:* Schema migration scripts execute cleanly; all query paths support indexed lookups by `job_id`.

---

### Epic 6: Asynchronous Queue & Celery Worker System
- **Epic Owner:** Sobia & Usman
- **Story 6.1: RabbitMQ & Celery Decoupling** (Owner: Sobia | Points: 5)
  - *Description:* Enforce asynchronous message routing over RabbitMQ AMQP; eliminate silent synchronous execution fallbacks.
  - *Acceptance Criteria:* Upload endpoint responds immediately (<100ms) with `status: QUEUED`; Celery worker consumes and processes the job.
- **Story 6.2: Resilience, Retries & Dead-Letter Queue** (Owner: Usman | Points: 5)
  - *Description:* Configure exponential backoff retries, task timeouts, and DLQ for failed messages.
  - *Acceptance Criteria:* Malformed or failing jobs are retried up to 3 times before routing to DLQ with detailed error logs.

---

### Epic 7: MLflow Tracking & S3 Model Artifact Management
- **Epic Owner:** Saim & Usman
- **Story 7.1: Centralized MLflow Tracking Server** (Owner: Saim | Points: 5)
  - *Description:* Configure MLflow server with S3 artifact backend for tracking parameters, metrics, PR curves, and model checkpoints.
  - *Acceptance Criteria:* MLflow UI displays active runs, metric charts, and versioned model artifacts.
- **Story 7.2: S3 Model Loading & Startup Integrity Check** (Owner: Usman | Points: 5)
  - *Description:* Implement automated model download from S3 with SHA-256 integrity verification during worker container startup.
  - *Acceptance Criteria:* Worker verifies hash against metadata, runs warm-up inference, and reports healthy status.

---

### Epic 8: Detection MLOps Retraining Automation
- **Epic Owner:** Usman & Junaid
- **Story 8.1: Hugging Face Dataset Revision Monitor** (Owner: Usman | Points: 5)
  - *Description:* Implement HMAC-authenticated webhook handler and revision polling with deduplication to detect new dataset commits.
  - *Acceptance Criteria:* New dataset revisions trigger automated pipeline; duplicate commits are rejected.
- **Story 8.2: Remote Retraining Trigger & Promotion Gate** (Owner: Junaid | Points: 8)
  - *Description:* Dispatch unattended training job on remote compute (AWS Batch / EC2 Spot), evaluate candidate against production model, and gate promotion.
  - *Acceptance Criteria:* Models exceeding mAP threshold are registered and deployed; regressions keep current champion active.

---

### Epic 9: AWS Cloud Infrastructure Deployment
- **Epic Owner:** Usman
- **Story 9.1: Multi-Stage Containerization & Amazon ECR** (Owner: Usman | Points: 5)
  - *Description:* Author optimized, non-root multi-stage Dockerfiles for API and worker; push to Amazon ECR.
  - *Acceptance Criteria:* Images scan clean of high/critical vulnerabilities; size optimized for fast deployment.
- **Story 9.2: ECS Fargate, RDS PostgreSQL & Networking** (Owner: Usman | Points: 8)
  - *Description:* Provision ECS Fargate services behind ALB, RDS PostgreSQL in private subnets, Amazon MQ RabbitMQ, and S3 buckets.
  - *Acceptance Criteria:* End-to-end cloud pipeline functional; health checks report 200 OK.

---

### Epic 10: Centralized Logging, Monitoring & Observability
- **Epic Owner:** Usman & Sobia
- **Story 10.1: Structured JSON Logging & Distributed Tracing** (Owner: Sobia | Points: 5)
  - *Description:* Implement structured JSON logger propagating `correlation_id` across FastAPI, RabbitMQ, Celery, and database.
  - *Acceptance Criteria:* Logs traceable end-to-end using single correlation ID.
- **Story 10.2: Prometheus Metrics & Grafana Dashboards** (Owner: Usman | Points: 5)
  - *Description:* Instrument Prometheus metrics (p50/p95/p99 latency, queue depth, active workers) and build Grafana dashboards.
  - *Acceptance Criteria:* Real-time operational dashboards display live request volumes and queue metrics.

---

### Epic 11: Performance, Stress & Cold-Start Testing
- **Epic Owner:** Usman & Saim
- **Story 11.1: Locust Concurrency Load Test Suite** (Owner: Usman | Points: 5)
  - *Description:* Author Locust test scenarios simulating 1, 5, 10, 20, and 50 concurrent document uploads to the deployed platform.
  - *Acceptance Criteria:* Real measured throughput, p95/p99 latency, and error rates captured in performance report.
- **Story 11.2: Cold-Start & Resource Saturation Profiling** (Owner: Saim | Points: 5)
  - *Description:* Measure container cold start, model download time, warm inference latency, and memory utilization under sustained load.
  - *Acceptance Criteria:* Cold-start and soak test report completed with bottleneck identification.

---

### Epic 12: Documentation, Architecture Diagrams & Final Integration
- **Epic Owner:** Usman & Team
- **Story 12.1: Cloud Architecture & Dataflow Diagrams** (Owner: Usman | Points: 3)
  - *Description:* Create production AWS-style architecture diagrams detailing inference and MLOps workflows.
  - *Acceptance Criteria:* High-resolution diagrams incorporated into system documentation.
- **Story 12.2: Final Runbook & Demonstration Suite** (Owner: All | Points: 5)
  - *Description:* Compile complete project documentation, deployment runbooks, API guides, and presentation slides.
  - *Acceptance Criteria:* Real, live, working platform demonstrated end-to-end with zero mock components.
