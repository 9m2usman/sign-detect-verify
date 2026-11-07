# 🖋️ Production Signature Detection & Verification Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![AWS Architecture](https://img.shields.io/badge/AWS-S3_%2B_SQS_%2B_ECS-orange.svg)](docs/architecture/AWS_ARCHITECTURE.md)
[![License Compliance](https://img.shields.io/badge/Compliance-Strictly_Non--Ultralytics-success.svg)](docs/compliance/LICENSE_AUDIT.md)

An end-to-end, commercial-grade platform for **Signature Object Detection**, **Signature Verification (1:1 Biometric Matching)**, and **Continuous MLOps Pipeline Automation**. 

Designed strictly with **permissive commercial open-source licenses (Apache 2.0, MIT, BSD)**, completely avoiding AGPL-licensed frameworks (e.g., Ultralytics YOLO).

---

## 🚀 Key Highlights & Architectural Decisions

* **Strict Commercial Licensing (Zero AGPL):** Built using **Hugging Face RT-DETR-R50** and **Megvii YOLOX-S** for detection, and **ResNet50 / ViT-B/16** for deep metric embeddings. See [`LICENSE_AUDIT.md`](docs/compliance/LICENSE_AUDIT.md).
* **Evidence-Based Model Selection:** Complete comparative experimentation reports:
  * [Stage A Detection Comparison (YOLOX-S vs. RT-DETR)](docs/model_selection/DETECTION_MODEL_COMPARISON.md)
  * [Stage B Verification Comparison (ResNet50 vs. ViT-B/16)](docs/model_selection/VERIFICATION_MODEL_COMPARISON.md)
  * [Final Production Decision Report](docs/model_selection/FINAL_MODEL_SELECTION.md)
* **Asynchronous AWS Architecture:** FastAPI staging to **Amazon S3** + **Amazon SQS** with an asynchronous **Worker Consumer**, Dead Letter Queue (DLQ) retry policies, and zero binary bloat in queue payloads.
* **Continuous MLOps Training Loop:** Hugging Face dataset webhook triggers $\rightarrow$ remote GPU training $\rightarrow$ **Comet ML / MLflow** experiment tracking $\rightarrow$ S3 model registry $\rightarrow$ worker zero-downtime hot-reload.

---

## 🏗️ System Architecture Flow

```
                           [ HTTPS Client / Web UI ]
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
                    |       Fetch Images    | - RT-DETR-R50         |
                    |       & Write Results | - ResNet50 Embedding  |
                    |                       +-----------------------+
                    |                                   |
                    |                                   | 4. Update Status
                    |                                   v
                    +---------------------- [ DynamoDB / S3 Results ]
```

---

## 🛠️ Quickstart & Local Execution

### 1. Installation
Clone the repository and install the verified permissive dependencies:
```bash
git clone https://github.com/9m2usman/sign-detect-verify.git
cd sign-detect-verify
pip install -r requirements.txt
```

### 2. Run Local Cloud Infrastructure (LocalStack Mock)
Spins up simulated Amazon S3 and SQS locally:
```bash
docker-compose up -d localstack
```

### 3. Launch FastAPI Ingestion Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation available at: `http://localhost:8000/docs`

### 4. Launch Asynchronous Worker Service
```bash
python -m worker.consumer
```

### 5. Run the Automated Test Suite
```bash
pytest tests/
```

---

## 📁 Repository Layout

```
├── api/                           # FastAPI ingestion endpoints & Pydantic schemas
├── ml/                            # Detection (RT-DETR/YOLOX) & Verification (ResNet/ViT)
├── worker/                        # SQS consumer worker service
├── training/                      # GPU training scripts & Colab notebooks
│   ├── detection/                 # YOLOX & RT-DETR training scripts
│   ├── verification/              # ResNet50 & ViT metric learning scripts
│   └── remote_gpu_training.ipynb  # Interactive Colab GPU training notebook
├── tests/                         # Automated unit and integration test suite
├── docs/                          # Comprehensive technical documentation
│   ├── architecture/              # AWS architecture & infrastructure specs
│   ├── model_selection/           # Comparative evaluation reports & final decisions
│   ├── dataset/                   # Dataset forensic audits & split analyses
│   └── compliance/                # License audits & commercial permission records
├── docker-compose.yml             # LocalStack S3 + SQS simulation
├── requirements.txt               # Strict Apache 2.0 / MIT dependency manifest
└── README.md
```

---

## ⚖️ Open-Source License
Distributed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for details.  
Full third-party compliance verified in [`docs/compliance/LICENSE_AUDIT.md`](docs/compliance/LICENSE_AUDIT.md).
