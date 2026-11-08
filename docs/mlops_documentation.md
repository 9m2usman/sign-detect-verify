# MLOps Architecture & Automated Retraining Documentation

**Module:** Machine Learning Operations (MLOps) Lifecycle  
**Tracker:** MLflow  
**Trigger:** Hugging Face Hub Dataset Webhooks  
**Execution:** Remote GPU Compute (Google Colab / AWS Batch GPU)

---

## 1. End-to-End MLOps Pipeline Flow

```text
Hugging Face Dataset Update Event
    │
    ▼
FastAPI Webhook Receiver (/api/v1/webhooks/huggingface)
    │  [HMAC-SHA256 Cryptographic Secret Validation]
    │  [Idempotency Guard: Avoid duplicate execution]
    ▼
Message Broker: training_queue (Amazon MQ / RabbitMQ)
    │
    ▼
Remote GPU Training Service (AWS Batch / Google Colab)
    │  1. Ingest authorized dataset revision commit
    │  2. Run data validation & COCO conversion
    │  3. Train candidate model (RTMDet / Siamese CNN)
    │  4. Run automated test set evaluation
    ▼
MLflow Experiment Tracking Server
    │  [Log mAP@50, mAP@50:95, EER, FAR, FRR, loss curves]
    │  [Save model weights to S3 Model Registry]
    ▼
Model Quality Gate (mlops/registry/model_registry.py)
    │  Is Candidate Metric > Champion Metric?
    ├── YES ──> Promote to Active Production Model -> Deploy
    └── NO  ──> Reject Candidate -> Alert CloudWatch -> Retain Champion
```

---

## 2. Model Experiment Tracking with MLflow

All detection and verification training runs are logged to MLflow via `mlops/tracking/mlflow_tracker.py`:

### Parameters Logged
- `model_architecture`: `rtmdet_tiny`, `yolox_tiny`, `siamese_resnet18`, `triplet_resnet18`
- `learning_rate`: e.g. `0.001`
- `weight_decay`: e.g. `0.0001`
- `batch_size`: e.g. `32`
- `dataset_revision_sha`: Git commit hash of dataset
- `hardware_gpu`: e.g. `Tesla T4 (15.0 GB)`

### Metrics Logged
- **Detection:** `loss`, `mAP@50`, `mAP@50:95`, `precision`, `recall`, `inference_latency_ms`
- **Verification:** `contrastive_loss`, `triplet_loss`, `eer`, `far`, `frr`, `roc_auc`, `accuracy`

---

## 3. Automated Model Registry & Promotion Criteria

The automated quality gate (`ModelRegistryGate`) enforces strict conditions:

1. **Detection Candidate Gate:**
   - Candidate mAP@50 must exceed **0.85** minimum baseline.
   - Candidate mAP@50 must exceed the current production champion's mAP@50.
2. **Verification Candidate Gate:**
   - Candidate Equal Error Rate (EER) must be below **0.08** (8%).
   - Candidate EER must be lower than the active champion's EER.
3. **Automated Rollback:**
   - If a promoted model fails post-deployment health checks or latency benchmarks, the worker automatically re-points to the prior champion artifact in S3.
