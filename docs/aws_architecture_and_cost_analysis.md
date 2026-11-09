# AWS Architecture & Financial Risk Management

**Document Version:** 1.0.0  
**Date:** November 2026  
**Project:** Signature Detection & Verification Platform  

---

## 1. System Architecture Overview

The platform uses an asynchronous, event-driven microservices architecture deployed on Amazon Web Services (AWS):

```mermaid
flowchart TD
    User["User / Client Application"] -->|HTTPS Upload| ALB["Application Load Balancer"]
    ALB -->|Forward Request| ECS_API["Amazon ECS Fargate (FastAPI API)"]
    
    ECS_API -->|Store Metadata| RDS["Amazon RDS PostgreSQL"]
    ECS_API -->|Store Raw Document| S3_Docs["Amazon S3 (Documents & Crops)"]
    ECS_API -->|Publish Job Message| AMQ["Amazon MQ / ECS (RabbitMQ Broker)"]
    
    AMQ -->|Consume Job| ECS_Worker["Amazon ECS Fargate (Celery Workers)"]
    S3_Models["Amazon S3 (Versioned Models)"] -->|Download Approved Weights| ECS_Worker
    S3_Docs -->|Retrieve Document| ECS_Worker
    
    ECS_Worker -->|Execute Detection| Det["Signature Detection (Faster R-CNN)"]
    Det -->|Extract Crops| Crop["Signature Cropping"]
    Crop -->|Execute Verification| Ver["Signature Verification (Siamese ResNet)"]
    
    Ver -->|Write Results| RDS
    Crop -->|Save Isolated Crops| S3_Docs
    
    ECS_API -->|Poll Status / Results| RDS
    ECS_API -->|Return Verification Decision| User
```

---

## 2. Detection MLOps Retraining Flow

```mermaid
flowchart TD
    HF["Hugging Face Detection Dataset (New Revision)"] -->|Webhook / Polling| Webhook["FastAPI /api/v1/webhooks/huggingface"]
    Webhook -->|Verify HMAC & Deduplicate| Dispatcher["Retraining Dispatcher"]
    Dispatcher -->|Launch Ephemeral Spot Instance| EC2_GPU["AWS EC2 Spot GPU (g4dn.xlarge)"]
    
    EC2_GPU -->|Pull Dataset & Train| Train["Detection Training (train_detector.py)"]
    Train -->|Log Metrics & Curves| MLflow["MLflow Tracking Server (with S3 Backend)"]
    Train -->|Evaluate on Test Split| Eval["Evaluation Gate (mAP@0.5, mAP@0.5:0.95)"]
    
    Eval -->|Pass Acceptance Gate| Reg["Model Registry Promotion"]
    Eval -->|Fail Acceptance Gate| Reject["Reject Candidate & Alert"]
    
    Reg -->|Upload Checkpoint| S3_Models["Amazon S3 (Approved Model Weights)"]
    Reg -->|Trigger Rolling Update| ECS_Worker["Inference Workers (Rolling Reload)"]
    EC2_GPU -->|Self-Terminate Instance| Term["Automatic Instance Shutdown"]
```

---

## 3. Comprehensive AWS Cost Risk Analysis

| AWS Service | Deployment Tier | Estimated Monthly Cost | Financial Risk Level | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **EC2 GPU Training (g4dn.xlarge)** | Retraining (Spot) | ~$0.526/hour (~$5 - $15 / run) | **HIGH (if left running)** | **Never use persistent GPU instances.** Use transient EC2 Spot instances or Google Colab with script auto-shutdown (`sudo shutdown -h now`). |
| **NAT Gateway** | VPC Networking | ~$32.40/month + $0.045/GB transfer | **HIGH (Common silent cost)** | Run development ECS tasks in public subnets with Internet Gateway and security groups. Deploy a single NAT Gateway only in final staging. |
| **Amazon MQ (RabbitMQ)** | Message Broker | ~$25 - $35/month (`mq.t3.micro`) | **MEDIUM** | Use single-AZ `mq.t3.micro` for staging, or run RabbitMQ as an ECS container task during development. |
| **Amazon RDS (PostgreSQL)** | Database | ~$15 - $25/month (`db.t4g.micro`) | **MEDIUM** | Enable auto-stop when not actively testing; use 20GB gp3 storage. |
| **Amazon ECS Fargate** | Compute (API + Workers) | ~$15 - $30/month (0.5 vCPU, 1 GB RAM) | **LOW - MEDIUM** | Use Fargate Spot (saves up to 70%); configure scaling to 0 tasks during idle hours. |
| **Amazon S3** | Object Storage | < $2.00/month | **VERY LOW** | Configure 14-day expiration lifecycle rules on test upload buckets. |
| **Amazon CloudWatch** | Monitoring & Logs | < $5.00/month | **LOW** | Set 7-day log retention policies to avoid log accumulation charges. |

**Total Estimated Deployment Running Cost:** ~$60 – $105/month during active testing (reducible to ~$15/month when idle or run via Docker on a single t3/t4g EC2 instance).
