# Continuous Integration & Continuous Deployment (CI/CD) Guide

## 1. Overview
The CI/CD pipeline guarantees high code quality, zero-license violations, automated regression testing, Docker container verification, and production artifact delivery on every Git event.

---

## 2. GitHub Actions Workflow Architecture (`.github/workflows/ci.yml`)

The pipeline executes five automated stages on every commit to `main` and all pull requests:

```
[git push / PR]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Containerized Service Spin-up                            │
│    - PostgreSQL 16 Alpine (:5432) with pg_isready check     │
│    - RabbitMQ 3.13 Alpine (:5672) with ping diagnostics     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Python 3.11 Setup & Pip Cache Rehydration                │
│    - Fast caching of pip packages                           │
│    - Installation of locked dependencies                   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Automated License Audit & Zero-AGPL Gatekeeper           │
│    - Enforces NO Ultralytics, GPL, or AGPL libraries        │
│    - Fails pipeline immediately if AGPL is detected         │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Comprehensive Pytest Test Suite                          │
│    - Unit Tests (Models, Preprocessing, Metrics, API)       │
│    - Integration Tests (Celery Tasks, DB Session, Pipeline) │
│    - End-to-End Test Execution (Full verify pipeline)       │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Multi-Stage Docker Image Build & Vulnerability Scan      │
│    - Validates Backend Dockerfile build                     │
│    - Validates Worker Dockerfile build                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Strict License Audit Step

Because AGPL-3.0 licenses contaminate commercial codebases and violate our project requirements, the CI pipeline enforces a programmatic check:

```bash
# Executed in GitHub Actions Runner:
python -c "import sys; 'ultralytics' in sys.modules and sys.exit(1) or print('Ultralytics cleanly absent.')"
pip list | grep -i ultralytics && exit 1 || echo "License check passed."
```

If any developer accidentally introduces an AGPL wheel, the PR is blocked automatically.

---

## 4. Test Execution & Coverage Thresholds

The pipeline runs `pytest` across all test suites:
```bash
DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/signature_verification_test_db \
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672// \
pytest -v --cov=backend/app --cov=worker --cov=mlops --cov-report=term-missing
```

---

## 5. Deployment Triggers (CD)

- **Push to `main` with passing tests**:
  - Automatically triggers Docker image build.
  - Pushes images to Amazon ECR tagged with Git commit SHA and `latest`.
  - Initiates ECS rolling deployment.
- **Hugging Face Dataset Webhook**:
  - Sends signed HMAC event to `/api/v1/webhooks/huggingface`.
  - Dispatches retraining pipeline job to Celery worker.
  - New model checkpoint is evaluated against the current champion model.
  - Promoted only if `val_loss` improves and mAP/ROC-AUC threshold is satisfied.
