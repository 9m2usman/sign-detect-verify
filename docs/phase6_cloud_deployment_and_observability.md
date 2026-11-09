# Phase 6: Cloud Deployment & Observability Architecture Report

## Executive Summary
Phase 6 delivers the production deployment topology, hardened multi-stage container images, automated Terraform Infrastructure as Code (IaC), and centralized observability for the **Signature Detection & Verification Platform**.
The platform is packaged into minimal, unprivileged container images (running as non-root `appuser:10001`), orchestrated via **Docker Compose** locally, mapped to **AWS ECS Fargate, RDS PostgreSQL, and S3** for cloud production, and continuously monitored via **Prometheus** metrics and **Grafana** dashboards.

---

## 1. Cloud Architecture & Service Mapping

```
                                  INTERNET
                                     |
                                     v
                 +---------------------------------------+
                 |    AWS Application Load Balancer     |
                 |             (Port 80/443)             |
                 +-------------------+-------------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |        AWS ECS Fargate Cluster        |
                 |                                       |
                 |   +-------------------------------+   |
                 |   |  FastAPI Backend Microservice |   |
                 |   |  (Multi-stage Docker, Port 80)|   |
                 |   +---------------+---------------+   |
                 |                   |                   |
                 |      Stores Task  | Enqueues Job      |
                 |                   v                   |
                 |   +---------------+---------------+   |
                 |   |    Celery Worker Service      |   |
                 |   |  (Deep Learning Inference)    |   |
                 |   +---------------+---------------+   |
                 +---------+---------+---------+---------+
                           |                   |
               PostgreSQL  |                   | AMQP Task Queue
               Connection  v                   v
              +------------+------+      +-----+-------------+
              | AWS RDS PostgreSQL|      | Amazon MQ         |
              | (db.t4g.micro)    |      | (RabbitMQ Broker) |
              +-------------------+      +-------------------+
                           ^
                           | Document & Model Storage
                           v
              +------------+------+
              |   AWS S3 Bucket   |
              | (Encrypted AES256)|
              +-------------------+
```

---

## 2. Hardened Multi-Stage Containerization

All Dockerfiles follow industry security standards:
1. **Multi-Stage Builds**:
   - `Stage 1 (builder)`: Compiles C extensions and wheels in `/opt/venv`.
   - `Stage 2 (runtime)`: Copies only pre-built wheels onto `python:3.11-slim`, discarding compilers (`gcc`, `g++`, header files).
2. **Unprivileged Non-Root User**:
   - Runs exclusively as `appuser` (UID 10001, GID 10001) with restricted permissions.
3. **No Hardcoded Secrets**:
   - All credentials (`DATABASE_URL`, `CELERY_BROKER_URL`, `HF_WEBHOOK_SECRET`) injected via environment variables.
4. **Automated Container Healthchecks**:
   - Periodic curl probes against `http://localhost:8000/health` and Celery worker ping diagnostics.

---

## 3. Infrastructure as Code (Terraform)

Located in [`terraform/`](file:///E:/sign_detect-verify/terraform/):
- **[`main.tf`](file:///E:/sign_detect-verify/terraform/main.tf)**: VPC (`10.0.0.0/16`), public subnets across 2 AZs, Internet Gateway, route tables, and security groups.
- **[`s3.tf`](file:///E:/sign_detect-verify/terraform/s3.tf)**: S3 bucket with AES256 server-side encryption, versioning, and lifecycle expiration for temporary cropped signatures (30-day retention).
- **[`rds.tf`](file:///E:/sign_detect-verify/terraform/rds.tf)**: PostgreSQL 16 on `db.t4g.micro` with storage auto-scaling (20 GB to 100 GB gp3).
- **[`ecs.tf`](file:///E:/sign_detect-verify/terraform/ecs.tf)**: ECS Fargate cluster, task definitions (1 vCPU, 2 GB RAM), Application Load Balancer, and target group with `/health` checks.
- **[`outputs.tf`](file:///E:/sign_detect-verify/terraform/outputs.tf)**: Public ALB DNS name, S3 bucket ARN, and RDS connection endpoints.

---

## 4. Observability & Monitoring Suite

### 4.1 Prometheus Telemetry (`GET /metrics`)
Exposes live application metrics formatted for Prometheus:
- `http_requests_total`: Throughput by method, endpoint, and status code.
- `http_request_duration_seconds`: Response latency histogram (p50, p95, p99 percentiles).
- `signature_detections_per_document`: Distribution of detected signatures.
- `signature_similarity_score_distribution`: Cosine similarity distribution (0.0 to 1.0).
- `signature_decisions_total`: Cumulative counts of `GENUINE` vs `FORGED` decisions.
- `worker_jobs_processed_total`: Asynchronous jobs completed vs failed.

### 4.2 Grafana Provisioning & Dashboard
- **Datasource**: Pre-configured Prometheus datasource (`http://prometheus:9090`).
- **Dashboard Panels**:
  - HTTP Request Throughput (req/sec)
  - API Latency Percentiles (p50, p95, p99)
  - Authenticity Decision Ratio (Donut chart of Genuine vs Forged)
  - Cosine Similarity Score Heatmap / Distribution
  - Bounding Box Detections per Document

---

## 5. Phase 6 Acceptance Matrix
| Requirement | Status | Verification Evidence |
| :--- | :--- | :--- |
| Multi-stage Dockerfiles (API, Worker, Retraining) | ✅ Verified | `backend/Dockerfile`, `worker/Dockerfile`, `training/Dockerfile` |
| Non-root container user (UID 10001) | ✅ Verified | Configured in all Dockerfiles with chown on `/app/storage` |
| Local multi-container Docker Compose | ✅ Verified | `docker-compose.yml` with Postgres, RabbitMQ, MLflow, Prometheus, Grafana |
| AWS Infrastructure as Code | ✅ Verified | Complete Terraform manifests in `terraform/` |
| Prometheus metrics endpoint | ✅ Verified | `/metrics` tested with latency, throughput, and ML histograms |
| Grafana operational dashboard | ✅ Verified | `signature_verification_dashboard.json` provisioned |
| Unit and integration tests | ✅ Verified | 32/32 tests passing |
