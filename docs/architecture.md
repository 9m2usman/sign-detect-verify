# Enterprise AWS Architecture Documentation

**System:** Signature Detection, Verification & Automated MLOps Platform  
**Architecture Style:** Event-Driven, Microservices-Decoupled Asynchronous Processing  
**Cloud Provider:** Amazon Web Services (AWS)

---

## 1. Architectural Overview & Design Principles

The platform separates real-time user request ingestion from computational computer vision and metric learning workloads.

### Core Architectural Pillars
1. **Asynchronous Ingestion (Decoupled Compute):** Heavy deep learning inference (object detection and embedding comparisons) is decoupled from the web tier via a durable message broker (**RabbitMQ / Amazon MQ**), preventing HTTP gateway timeouts.
2. **Strict Licensing Enforcement:** Strictly zero copyleft (AGPL) dependencies. The entire vision stack relies on **Apache 2.0** (RTMDet-Tiny, YOLOX-Tiny) and **BSD-3-Clause** (PyTorch Siamese & Triplet networks).
3. **Automated Continuous Retraining (MLOps):** Changes to Hugging Face dataset repositories trigger authenticated webhook events, queueing remote GPU training jobs on **AWS Batch** or Google Colab, and logging metrics to an **MLflow** tracking server.
4. **Data Isolation & Security:** Storage buckets are encrypted at rest with AWS KMS. The FastAPI backend and Celery workers operate within private VPC subnets behind an Application Load Balancer (ALB).

---

## 2. Component Breakdown

### 2.1 Ingestion & Edge Tier
- **Amazon CloudFront:** Global Content Delivery Network (CDN) providing SSL/TLS termination, DDoS mitigation via AWS Shield, and static asset caching.
- **AWS WAF:** Protects against common web exploits (SQL injection, cross-site scripting, request flooding).
- **Application Load Balancer (ALB):** Distributes incoming HTTPS requests across Amazon ECS tasks in multiple Availability Zones.

### 2.2 Microservices & Worker Tier
- **FastAPI Service (Amazon ECS Fargate):** Lightweight asynchronous REST API exposing document upload, job submission, status polling, and webhook endpoints.
- **Message Broker (Amazon MQ for RabbitMQ):** Manages durable, persistent AMQP queues:
  - `inference_queue`: Document verification jobs dispatched by FastAPI.
  - `training_queue`: Retraining jobs triggered by dataset updates.
- **Celery Inference Worker (Amazon ECS Fargate):** Dedicated workers that consume jobs from `inference_queue`, execute `UnifiedSignaturePipeline`, and persist results to the database.

### 2.3 Storage & Persistence Tier
- **Amazon RDS PostgreSQL:** Relational store containing persistent job records, bounding box coordinates, confidence scores, and authenticity decisions.
- **Amazon S3 (Documents & Crops Bucket):** Encrypted object storage for original uploaded documents and isolated signature crop cutouts.
- **Amazon S3 (Model Registry Bucket):** Storage for approved model checkpoints (`.pth`, `.onnx`) tagged with model version and Git commit hash.

### 2.4 Automated Retraining & MLOps Tier
- **Hugging Face Hub Webhook:** Fires HTTP POST on dataset updates to `/api/v1/webhooks/huggingface`.
- **AWS Batch (EC2 GPU Instances - G4dn/G5):** Spin up on-demand GPU containers to run RTMDet and Siamese metric learning retraining pipelines.
- **MLflow Tracking Server:** Centralized logging of mAP@50, mAP@50:95, EER, loss curves, and artifact lineage.
- **Model Registry Quality Gate:** Compares candidate model metrics against active champion before automated blue/green deployment.
