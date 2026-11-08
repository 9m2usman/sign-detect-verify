# AWS Cloud Deployment Architecture & Runbook

## 1. Cloud Infrastructure Overview
The target production deployment is designed for high availability and elastic scalability on AWS:

- **Ingestion & CDN**: Amazon Route 53 DNS + AWS WAF + Amazon CloudFront.
- **Compute Cluster**: Amazon ECS (Elastic Container Service) with AWS Fargate.
  - Backend Service: Autoscaled based on CPU/Memory and HTTP target request count.
  - Worker Service: Autoscaled based on RabbitMQ queue depth metrics.
- **Message Broker**: Amazon MQ for RabbitMQ (multi-AZ broker).
- **Relational Database**: Amazon RDS for PostgreSQL 16 (Multi-AZ with automated backups).
- **Object Storage**: Amazon S3 buckets for raw documents, signature crops, and MLflow model artifacts.
- **Monitoring**: Amazon CloudWatch alarms, logs, and container insights.

## 2. Deployment Procedures
1. Push Docker images to Amazon ECR.
2. Update ECS task definitions with latest image digests.
3. Apply Terraform / CloudFormation stack updates.
4. Run database migrations via temporary ECS task.
5. Perform blue/green deployment cutover with zero downtime.
