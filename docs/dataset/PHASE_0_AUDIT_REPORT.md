# PHASE 0: Repository & Dataset Audit Report
**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Target GitHub Repository:** [https://github.com/9m2usman/sign-detect-verify](https://github.com/9m2usman/sign-detect-verify)  
**Execution Date:** 2026-11-07  
**Author:** Engineering Team (Senior ML & MLOps Lead)

---

## 1. Executive Summary

This report concludes **Phase 0** of the Master Execution Plan. A comprehensive forensic audit was performed across:
1. The local workspace directory (`E:\sign_detect-verify`) and Git state.
2. The existing dataset (`merged_signature_dataset/`) containing 28,958 images.
3. Verification label availability and signer identity distribution.
4. GPU training infrastructure options.
5. Permissive licensing (strict non-Ultralytics / non-AGPL compliance).

---

## 2. Current Repository Assessment

### 2.1 Workspace & Git State
* **Root Location:** `E:\sign_detect-verify`
* **Parent Drive Anomaly:** The root drive `E:\` contained a previous unrelated Git repository. Git commands executed in the root sandbox initially failed with `fatal: cannot change to 'E:/'`.
* **Remediation & Initialization:**
  - An empty Git repository was cleanly initialized inside `E:\sign_detect-verify`.
  - Default branch set to `main`.
  - Configured remote: `origin -> https://github.com/9m2usman/sign-detect-verify.git`.
  - Created `.gitignore` excluding datasets (`merged_signature_dataset/`, `*.zip`), model checkpoints (`*.pt`, `*.pth`), temporary test caches, and secrets (`.env`).
* **Existing Files in Repository Root:**
  - `PROJECT_PLAN_AND_ARCHITECTURE.md`: Master architectural blueprint & Jira stories.
  - `docker-compose.yml`: LocalStack mock infrastructure for AWS S3 and SQS.
  - `requirements.txt`: Curated dependency specification strictly avoiding AGPL libraries.
  - `.gitignore`: Production-grade exclusion rules.

---

## 3. Dataset Structure & Annotation Assessment

### 3.1 Overview & Split Verification
We programmatically inspected the entire directory tree of `merged_signature_dataset/`:

| Split | Image Files Count | Label Files Count | Bounding Boxes Count | Empty (Negative) Images | Sub-Image / Document Boxes | Full-Image Crops ($w,h > 0.9$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 22,953 | 22,953 | 29,952 | 1,106 | 28,092 | 1,860 |
| **Valid** | 3,287 | 3,287 | 4,670 | 99 | 4,592 | 78 |
| **Test** | 2,718 | 2,718 | 3,472 | 80 | 3,395 | 77 |
| **Total** | **28,958** | **28,958** | **38,094** | **1,285** | **36,079** | **2,015** |

### 3.2 Annotation Format
* **Configuration:** Defined in `merged_signature_dataset/data.yaml`
* **Format:** YOLO standard bounding box coordinates (`class cx cy w h`, normalized to $[0.0, 1.0]$).
* **Classes:** Exactly 1 class (`0: signature`). No multi-class contamination was found across any of the 28,958 label files.
* **Image Specifications:** Sample inspection confirmed 24-bit JPEG format, standard resolution $640 \times 640$ pixels.
* **Negative Samples:** 1,285 true negative document images containing no signatures (empty `.txt` files), which is essential to prevent false-positive signature hallucination on blank document regions.

---

## 4. Verification-Label Availability Assessment

### 4.1 Critical Finding
* **The `merged_signature_dataset/` contains ONLY bounding boxes for object detection.**
* It contains **NO signer identity metadata**, genuine/forgery labels, or paired genuine-forgery pairs.
* **Mathematical Justification:** Signature verification is a biometric 1:1 matching problem requiring classification or metric learning on:
  $$\mathcal{D}_{\text{verif}} = \{(x_i, x_j, y_{ij}) \mid y_{ij} \in \{0, 1\}\}$$
  where $y_{ij}=1$ denotes identical genuine signer and $y_{ij}=0$ denotes skilled forgery or random impostor. Attempting to train or evaluate a verification model (ResNet50 / ViT) without signer labels would produce mathematically invalid metrics and violate the integrity rules.

### 4.2 Solution: Approved Permissive Verification Benchmark
We identified high-quality, legally permissible benchmark datasets on Hugging Face Hub under the **MIT License**:
1. **Primary Candidate:** `rakshitdabral/Signature-Verification-Dataset` (and mirror `Peachyy2208/Signature-Verification-Dataset`)
   - **License:** **MIT License** (100% commercially permissible).
   - **Structure:** Signer-partitioned Genuine and Forgery samples (e.g., CEDAR / ICDAR style).
   - **Signer-Independent Protocol:** Disjoint signers for Training, Validation, and Testing to prevent identity leakage.

---

## 5. GPU Environment Options

Since the local development host is CPU-only, model training must be executed remotely:

1. **Option 1 (Primary): Google Colab GPU (T4 / V100 / A100)**
   - High availability, native Google Drive persistence for checkpoints.
   - Ideal for training YOLOX-S and RT-DETR.
2. **Option 2 (Secondary / Fallback): Kaggle GPU (2x T4 or P100)**
   - 30 hours/week free GPU quota, persistent dataset mounts.
3. **Local Machine (Windows 11 CPU):**
   - Strictly reserved for dataset auditing, API development, Docker orchestration, unit testing, and documentation.

---

## 6. Proposed Project Directory Structure

```
sign_detect-verify/
├── .github/
│   └── workflows/
│       ├── test_and_lint.yml          # CI test runner
│       └── hf_training_trigger.yml    # Continuous training dispatcher
├── api/                               # FastAPI Ingestion & Serving
│   ├── __init__.py
│   ├── main.py                        # Endpoint definitions
│   └── schemas.py                     # Pydantic request/response models
├── worker/                            # Asynchronous SQS Consumer
│   ├── __init__.py
│   └── consumer.py                    # Queue polling, DLQ & S3 updates
├── ml/                                # Inference Engines
│   ├── __init__.py
│   ├── detector.py                    # Unified YOLOX / RT-DETR detector
│   ├── verification.py                # ResNet50 & ViT metric embedding models
│   ├── engine.py                      # Unified SignatureEngine
│   └── utils.py                       # Preprocessing & bbox math
├── training/                          # GPU Training Scripts & Notebooks
│   ├── detection/
│   │   ├── train_yolox.py
│   │   ├── train_rtdetr.py
│   │   └── yolox_colab_notebook.ipynb
│   └── verification/
│       ├── train_resnet50.py
│       ├── train_vit.py
│       └── verification_colab_notebook.ipynb
├── tests/                             # Automated Test Suite
│   ├── test_dataset.py
│   ├── test_detector.py
│   ├── test_verification.py
│   ├── test_engine.py
│   └── test_api.py
├── docs/                              # Comprehensive Technical Docs
│   ├── model_selection/
│   ├── architecture/
│   └── compliance/
├── docker-compose.yml                 # LocalStack S3 + SQS local simulation
├── requirements.txt                   # Verified Apache 2.0 / MIT dependencies
└── README.md
```

---

## 7. Open-Source Dependency & Licensing Audit

* **Strict Prohibition:** Ultralytics (`ultralytics`), YOLOv5/v8/v11 (`AGPL-3.0`) are completely absent.
* **Detection Stack:**
  - **YOLOX:** Apache 2.0 (Megvii).
  - **RT-DETR:** Apache 2.0 (Baidu / Hugging Face `transformers`).
* **Verification Stack:**
  - **ResNet50 / ViT-B/16:** BSD 3-Clause (`torchvision` / `timm`).
* **Experiment Tracking:**
  - **Comet ML** (Apache 2.0 SDK) or **MLflow** (Apache 2.0).

---

## 8. Identified Risks & Mitigation Matrix

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Colab/Kaggle Session Timeout** | High | Checkpoint saving to Google Drive / Kaggle output every epoch with resume flags. |
| **Identity Leakage in Verification** | High | Strict signer-independent partitioning (Signers in Test set never appear in Train/Val). |
| **SQS 256KB Payload Limit** | Critical | SQS only transfers S3 object pointers (`s3://...`), never raw image bytes. |
| **License Contamination** | Critical | Regular automated dependency tree audits documented in `docs/compliance/LICENSE_AUDIT.md`. |
