# AWS Cloud Production Deployment Guide & Runbook

## 1. Enterprise Architecture Overview

The Signature Detection & Verification platform is deployed on AWS using an enterprise-grade, highly available, secure, and elastically scalable microservices pattern:

```
[Client / UI / Partner API]
            │ (HTTPS :443)
            ▼
    [Amazon Route 53]
            │
            ▼
       [AWS WAF] (Rate limiting, SQLi, XSS, IP reputation)
            │
            ▼
 [Application Load Balancer (ALB)] (Multi-AZ SSL Termination)
            │
    ┌───────┴────────────────────────┐
    │                                │
    ▼                                ▼
[ECS Fargate: Backend API]   [ECS Fargate: Model Workers]
(FastAPI, 2-10 Tasks)        (Celery, 2-20 Tasks autoscale on Queue Depth)
    │                                ▲
    │ (Enqueue async jobs)           │ (Consume jobs)
    ▼                                │
[Amazon MQ: RabbitMQ 3.13] ──────────┘
(Multi-AZ Active/Standby)
    │
    ├────────────────────────────────┐
    ▼                                ▼
[Amazon RDS PostgreSQL 16]   [Amazon S3 Buckets]
(Multi-AZ, Encrypted KMS)    (Documents, Crops, Model Weights)
                                     ▲
                                     │
                             [AWS SageMaker]
                             (GPU Spot Training Jobs triggered via Webhook)
```

---

## 2. Infrastructure Components & Specifications

| Tier | AWS Service | Specification / Instance Type | High Availability Strategy |
| :--- | :--- | :--- | :--- |
| **DNS & Edge** | Route 53 + AWS WAF | Latency-based routing + Managed Core Rule Set | Global Anycast |
| **Ingress** | Application Load Balancer | Internet-facing, dual-stack, HTTPS ACM cert | Multi-AZ (3 Availability Zones) |
| **API Compute** | Amazon ECS on AWS Fargate | 1 vCPU, 2 GB RAM per task (Min: 2, Max: 10) | Multi-AZ spread constraint |
| **Worker Compute**| Amazon ECS on AWS Fargate | 2 vCPU, 4 GB RAM per task (Min: 2, Max: 20) | Target Tracking on Queue Depth |
| **Message Broker**| Amazon MQ (RabbitMQ) | `mq.m5.large` Active/Standby cross-AZ pair | Automatic failover (< 30s) |
| **Database** | Amazon RDS PostgreSQL | `db.r6g.xlarge` (Multi-AZ with Read Replica) | Synchronous block-level replication|
| **Object Store** | Amazon S3 | S3 Standard with Glacier Lifecycle Rules | 99.999999999% (11 9s) durability |
| **Container Registry**| Amazon ECR | Private repository with KMS encryption | Multi-region replication |
| **Secrets & Config** | AWS Secrets Manager | Automatic rotation enabled | KMS-CMK encrypted |
| **Observability** | CloudWatch + Container Insights | Metric alarms, structured log groups, X-Ray traces | 30-day retention |

---

## 3. Step-by-Step AWS Deployment Procedures

### Step 1: ECR Repositories Setup & Image Push
```bash
# 1. Authenticate Docker with Amazon ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 2. Create Repositories
aws ecr create-repository --repository-name sign-verify/backend --image-scanning-configuration scanOnPush=true
aws ecr create-repository --repository-name sign-verify/worker --image-scanning-configuration scanOnPush=true

# 3. Build & Tag Multi-stage Docker Images
docker build -t sign-verify/backend:v1.0.0 -f backend/Dockerfile .
docker tag sign-verify/backend:v1.0.0 <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sign-verify/backend:v1.0.0

docker build -t sign-verify/worker:v1.0.0 -f worker/Dockerfile .
docker tag sign-verify/worker:v1.0.0 <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sign-verify/worker:v1.0.0

# 4. Push to ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sign-verify/backend:v1.0.0
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sign-verify/worker:v1.0.0
```

### Step 2: Amazon RDS PostgreSQL & Amazon MQ Provisioning
```bash
# Provision Amazon RDS PostgreSQL 16 Multi-AZ
aws rds create-db-instance \
    --db-instance-identifier sign-verify-prod-db \
    --db-instance-class db.r6g.xlarge \
    --engine postgres \
    --engine-version 16.2 \
    --master-username signadmin \
    --master-user-password <STRONG_GENERATED_PASSWORD> \
    --allocated-storage 100 \
    --multi-az \
    --storage-encrypted \
    --vpc-security-group-ids sg-0123456789abcdef0 \
    --no-publicly-accessible

# Provision Amazon MQ RabbitMQ Broker
aws mq create-broker \
    --broker-name sign-verify-rabbitmq \
    --engine-type RABBITMQ \
    --engine-version 3.13 \
    --deployment-mode ACTIVE_STANDBY_MULTI_AZ \
    --host-instance-type mq.m5.large \
    --users Username=rabbitadmin,Password=<STRONG_BROKER_PASSWORD>
```

### Step 3: ECS Fargate Task Definitions & Service Deployment
Task definitions reference AWS Secrets Manager for sensitive credentials:
- `DATABASE_URL`: `postgresql+psycopg2://signadmin:<PASS>@sign-verify-prod-db.xyz.us-east-1.rds.amazonaws.com:5432/sign_verify`
- `CELERY_BROKER_URL`: `amqp://rabbitadmin:<PASS>@b-xyz-1.mq.us-east-1.amazonaws.com:5671//`
- `S3_STORAGE_BUCKET`: `sign-verify-documents-prod`
- `MODEL_WEIGHTS_PATH`: `s3://sign-verify-model-vault/checkpoints/`

Deploy with Blue/Green rolling cutover:
```bash
aws ecs update-service \
    --cluster sign-verify-cluster \
    --service backend-service \
    --force-new-deployment \
    --deployment-configuration maximumPercent=200,minimumHealthyPercent=100
```

---

## 4. Worker Autoscaling Based on Queue Depth

Workers scale automatically using a custom CloudWatch metric published from RabbitMQ queue depth:
- **Scale Out**: When `MessagesInQueue > 50` for 2 minutes $\to$ Add 2 worker tasks.
- **Scale In**: When `MessagesInQueue == 0` for 10 minutes $\to$ Remove 1 worker task (down to minimum 2 tasks).

---

## 5. Security & Compliance Checklist
- [x] Zero AGPL/GPL dependencies in Docker images.
- [x] Non-root execution (`USER appuser`) inside containers.
- [x] Private subnets with NAT Gateways for ECS tasks; no public IP allocation.
- [x] RDS and S3 encrypted with AWS KMS customer-managed keys (CMK).
- [x] TLS 1.3 enforced on ALB and RabbitMQ (AMQPS :5671).
