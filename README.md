# Document Signature Detection and Verification System

Production-grade, permissively licensed (Apache 2.0 / MIT) document signature detection and offline verification pipeline.

---

## 1. System Overview

This system provides enterprise-grade signature extraction from arbitrary document layouts (invoices, contracts, identity documents) and pairwise signature verification to detect skilled and unskilled forgeries.

Key capabilities:
- **Signature Detection**: Permissive object detection (Faster R-CNN / Torchvision ResNet-FPN) strictly avoiding AGPL/Ultralytics restrictions.
- **Signature Verification**: Deep Siamese metric learning network (ResNet backbone with contrastive / triplet loss and cosine similarity distance).
- **Asynchronous Architecture**: Scalable task processing using Celery backed by RabbitMQ message broker.
- **Service Tier**: Modern asynchronous REST API built with FastAPI and Pydantic v2.
- **MLOps & Tracking**: Experiment logging, model checkpoint tracking, and model registry via MLflow.
- **Relational Persistence**: Job status, document metadata, and writer profiles managed in PostgreSQL 16.

---

## 2. Repository Layout

```
├── backend/                  # FastAPI REST API microservice
│   ├── app/
│   │   ├── api/              # Route handlers & API endpoints
│   │   ├── core/             # Configuration & security settings
│   │   ├── database/         # SQLAlchemy models and session lifecycle
│   │   ├── schemas/          # Pydantic v2 request/response schemas
│   │   ├── services/         # Business logic and dispatch services
│   │   └── models/           # Domain entities
│   ├── tests/                # Backend API tests
│   └── Dockerfile
├── worker/                   # Celery asynchronous inference worker
│   ├── tasks/                # Celery background tasks (detect, verify)
│   ├── services/             # Pipeline inference execution engines
│   ├── celery_app.py         # Celery instance configuration
│   └── Dockerfile
├── detection/                # Signature detection module (Faster R-CNN)
│   ├── configs/              # Hyperparameter and model configuration
│   ├── preprocessing/        # Document resizing, normalization, tiling
│   ├── training/             # PyTorch training loops & loss functions
│   ├── evaluation/           # Evaluation metrics (mAP@0.5, mAP@0.5:0.95)
│   ├── experiments/          # Model comparison experiments
│   └── selected_model/       # Selected production checkpoint reference
├── verification/             # Signature verification module (Siamese ResNet)
│   ├── preprocessing/        # Signature crop normalization & alignment
│   ├── models/               # Siamese network architectures
│   ├── training/             # Contrastive/triplet training pipelines
│   ├── evaluation/           # ROC-AUC, EER, and accuracy evaluation
│   ├── experiments/          # Benchmark experiments (CEDAR, BHSig)
│   └── selected_model/       # Selected production verification model
├── notebooks/                # Exploratory and experimental notebooks
├── mlops/                    # Tracking, registry, and pipeline automation
├── infrastructure/           # Docker, AWS ECS/CloudFormation, GitHub Actions
├── tests/                    # Global unit, integration, and E2E test suites
├── data/                     # Data documentation & local processing paths
├── docs/                     # Comprehensive engineering & research documentation
├── architecture/             # Architecture diagrams (Mermaid format)
├── docker-compose.yml        # Multi-container local orchestration
├── requirements.txt          # Permissively licensed dependencies
└── LICENSES.md               # Legal compliance & open-source license audit
```

---

## 3. Strict License Compliance

All models, frameworks, and third-party libraries incorporated into this project have been audited under permissive terms (**MIT**, **Apache 2.0**, **BSD-3-Clause**). 
Ultralytics (GPL / AGPL-3.0) and other copyleft licenses are **strictly excluded**. See [`LICENSES.md`](file:///e:/sign_detect-verify/LICENSES.md) and [`docs/model_licensing.md`](file:///e:/sign_detect-verify/docs/model_licensing.md) for full compliance audits and citations.

---

## 4. Getting Started

### Local Setup with Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/example/sign_detect-verify.git
cd sign_detect-verify

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install permissive dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running with Docker Compose

To spin up the complete local stack (PostgreSQL, RabbitMQ, MLflow, FastAPI backend, Celery worker):

```bash
docker-compose up -d
```

- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking Dashboard**: [http://localhost:5000](http://localhost:5000)
- **RabbitMQ Management Console**: [http://localhost:15672](http://localhost:15672) (guest / guest)
