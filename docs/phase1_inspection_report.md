# Phase 1: Repository Inspection, Forensic Audit & Architecture Baseline

**Document Version:** 1.0.0  
**Date:** November 2026  
**Project:** Signature Detection & Verification Platform  
**Repository:** `E:\sign_detect-verify`  

---

## 1. Executive Summary

This report documents the forensic baseline inspection of the Signature Detection and Verification platform repository. The audit covered repository topology, dataset separation, code integrity, ML model weights, execution environments, and cloud infrastructure requirements.

### Key Inspection Results:
1. **Repository Topology:**
   - Active code, checkpoints, datasets, and container configs reside in `E:\sign_detect-verify`.
   - Upstream project requirements and specifications are maintained in `E:\signature-ai`.
   - `E:\detect-verify` is the designated clean root workspace.
2. **Dataset Separation (Critical Gate):**
   - **Dataset A (Detection Only):** Fully verified at `E:\sign_detect-verify\merged_signature_dataset\` with **28,958 images** and **38,094 bounding boxes** in YOLO format. Contains zero writer identities and zero genuine/forged labels.
   - **Dataset B (Verification Only):** Not yet acquired. Strict separation maintained. Awaiting Phase 2 evaluation of CEDAR vs. BHSig260.
3. **Critical Defects Identified in Codebase:**
   - **Negative Sample Corruption:** `training/detection/train_rtdetr.py` (lines 78–81) converts background negative images into full-page bounding boxes labeled as signatures (`labels=[1]`).
   - **Undertrained Detection Checkpoint:** `checkpoints/rtdetr_r50/detector_best.pth` achieves max confidence of `0.3635` on document tests, causing 100% of neural detections to be discarded by the 0.40 threshold, silently triggering an ink heuristic with hardcoded 0.95 confidence.
   - **Mock Verification Training:** `training/verification/train_resnet50.py` trains on synthetic `torch.rand()` tensors instead of real signature pairs.
   - **Miscalibrated Verification Heuristic:** `verify_document.py` falsely rejects authentic reference signatures (score 0.4705 vs. threshold 0.82).

---

## 2. Directory Layout & Existing Implementation

```text
E:\sign_detect-verify\
├── backend/                             # FastAPI microservice
│   ├── app/
│   │   ├── api/v1/endpoints/           # jobs.py, models.py, signatures.py, webhooks.py
│   │   ├── core/config.py              # Pydantic v2 Settings (SQLite dev / PostgreSQL prod)
│   │   ├── database/                   # SQLAlchemy engine and session
│   │   ├── models/job.py               # Job and VerificationResult entities
│   │   ├── schemas/jobs.py             # Pydantic schemas
│   │   └── static/index.html           # Full HTML5 UI dashboard (770 lines, Tailwind CSS)
│   ├── Dockerfile
│   └── tests/                          # 35 unit/integration tests
├── detection/                           # Detection package
│   ├── preprocessing/                  # yolo_to_coco.py, audit_dataset.py
│   └── selected_model/detector.py      # ProductionSignatureDetector (Faster R-CNN + heuristic fallback)
├── verification/                        # Verification package
│   ├── models/                         # siamese_cnn.py, triplet_network.py
│   ├── preprocessing/                  # signature_cropper.py, preprocess.py
│   ├── evaluation/                     # evaluate_verification.py
│   └── selected_model/verifier.py      # Handcrafted biometric heuristic
├── worker/                              # Celery worker service
│   ├── celery_app.py                   # Celery broker configuration
│   ├── Dockerfile
│   └── tasks/inference_tasks.py        # Asynchronous job processor
├── mlops/                               # MLOps orchestration
│   ├── pipeline.py                     # UnifiedSignaturePipeline
│   ├── tracking/mlflow_tracker.py      # MLflow tracker with local JSON fallback
│   ├── registry/model_registry.py      # Quality gates (min_map50, max_eer)
│   └── webhooks/hf_handler.py          # HMAC verification for HF webhooks
├── checkpoints/
│   └── rtdetr_r50/detector_best.pth    # 165.7 MB PyTorch state dict (Faster R-CNN ResNet-50)
├── merged_signature_dataset/            # Extracted Dataset A (57,918 files)
├── merged_signature_dataset.zip         # 3.50 GB raw archive
├── training/                            # Training scripts
│   ├── detection/train_rtdetr.py       # Trains on YOLO format
│   └── verification/train_resnet50.py  # Mock Siamese training on random tensors
├── notebooks/                           # Jupyter notebooks
│   ├── user_colab_notebook.ipynb       # 11-cell end-to-end Colab training runner
│   ├── detection/train_rtmdet.ipynb
│   └── verification/train_siamese.ipynb
├── sample_test_run/                     # sample_document.jpg, reference_authentic.jpg, reference_impostor.jpg
├── docker-compose.yml                   # PostgreSQL 16, RabbitMQ 3.13, MLflow 2.13, API, Worker
├── requirements.txt                     # Python dependencies
├── verify_document.py                   # Standalone CLI tool
├── signature_dev.db                     # Local SQLite database (53 KB)
└── LICENSES.md                          # Permissive license audit
```

---

## 3. Dataset A (Detection) Audit Metrics

| Metric | Train Split | Validation Split | Test Split | Total Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Image Count** | 22,953 | 3,287 | 2,718 | **28,958** |
| **Label Count** | 22,953 | 3,287 | 2,718 | **28,958** |
| **Images with Signatures**| 21,847 | 3,188 | 2,638 | **27,673** |
| **True Negative Backgrounds**| 1,106 | 99 | 80 | **1,285** |
| **Bounding Box Count** | 29,952 | 4,670 | 3,472 | **38,094** |
| **Resolution** | 640×640 | 640×640 | 640×640 | **640×640 JPEG** |
| **Class Label** | `0 (signature)` | `0 (signature)` | `0 (signature)` | Exactly 1 class |

---

## 4. Architectural Conflicts & Technical Debt

1. **Model Architecture Naming Inconsistency:**
   - Directory is named `rtdetr_r50`, but the model checkpoint is actually a Torchvision `fasterrcnn_resnet50_fpn`.
   - *Resolution:* Formally standardize naming on **Torchvision Faster R-CNN ResNet-50 FPN** (BSD-3-Clause) or **MMDetection RTMDet-Tiny** (Apache 2.0).
2. **Ephemeral Disk vs. Shared S3 Object Storage:**
   - FastAPI saves uploaded files to local disk `./uploads/{job_id}/`.
   - In containerized Docker Compose or AWS ECS, workers cannot access files across separate containers.
   - *Resolution:* Configure shared volume `uploads_data` for local Docker Compose and stream to **Amazon S3** via `boto3` in production.
3. **Queue Fallback Masking Broker Outages:**
   - `signatures.py` catches broker errors and runs synchronously.
   - *Resolution:* Disallow silent synchronous execution in production to ensure queue reliability and health visibility.
4. **Unused Redis Dependency:**
   - `requirements.txt` contains `redis>=5.0.4`, but broker and Celery use RabbitMQ AMQP RPC.
   - *Resolution:* Remove `redis` to eliminate unnecessary dependencies.

---

## 5. Phase 1 Acceptance Gate

Phase 1 audit is complete. All findings, metrics, and risk assessments are verified. Proceeding to Phase 2 requires user confirmation on:
- Workspace directory mapping (`E:\detect-verify` vs `E:\sign_detect-verify`).
- Detection model framework selection (Torchvision Faster R-CNN vs MMDetection RTMDet-Tiny).
- Verification dataset candidates (CEDAR vs BHSig260).
