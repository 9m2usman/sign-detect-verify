# Continuous MLOps Pipeline & Automated Model Training Specification

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Target Environment:** AWS Cloud / Hugging Face Hub / GitHub Actions / Comet ML / MLflow

---

## 1. MLOps Architecture Overview

The MLOps pipeline automates the lifecycle of model training, evaluation, validation quality gates, artifact versioning, and zero-downtime deployment updates.

```
+---------------------------------------------------------------------------------+
|                                CONTINUOUS MLOps LOOP                            |
+---------------------------------------------------------------------------------+
   1. Data Ingestion & Versioning
      [ Hugging Face Dataset ] --(Webhook on commit)--> [ GitHub Actions Runner ]
                                                                   |
   2. Remote GPU Training Dispatch                                 v
      [ GitHub Actions ] --(Dispatches Job)---------> [ AWS SageMaker / GPU Node ]
                                                                   |
   3. Experiment Tracking                                          v
      [ Remote Training Job ] --(Logs loss, mAP, EER)-> [ Comet ML / MLflow ]
                                                                   |
   4. Validation Quality Gate                                      v
      [ Quality Gate Check ] <--(mAP >= 0.95 & EER <= 0.08)--------+
              |                                                    |
              | Passed                                             | Failed
              v                                                    v
   5. Model Artifact Promotion                              [ Pipeline Halts ]
      [ S3 Model Registry ]                                 [ Alert Sent ]
      (s3://sig-model-registry/vX.Y.Z/)
              |
              v
   6. Zero-Downtime Rollout
      [ SQS Broadcast / Worker Hot-Reload ]
```

---

## 2. Automated Training Trigger Flow

1. **Trigger Event:** Data annotators or contributors commit new verified document samples to the designated Hugging Face Dataset repository (`rakshitdabral/Signature-Verification-Dataset` or team private mirror).
2. **Webhook Ingestion:** Hugging Face Webhook dispatches an HTTP POST payload with commit SHA and dataset version tags to GitHub Actions repository dispatch.
3. **Workflow Execution:** [`.github/workflows/hf_training_trigger.yml`](file:///e:/sign_detect-verify/.github/workflows/hf_training_trigger.yml) initiates:
   - Validates permissive licensing compliance (strictly excludes AGPL/Ultralytics).
   - Launches the remote training runner on AWS SageMaker Training Jobs or EC2 GPU runners.

---

## 3. Experiment Tracking Protocol (Comet ML & MLflow)

Every remote training iteration logs:
* **Hyperparameters:** Learning rate, batch size, weight decay, backbone architecture, image resolution ($640 \times 640$ for detection, $224 \times 224$ for verification).
* **Epoch-by-Epoch Metrics:** Training loss, validation loss, mAP@50, mAP@50:95, False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER).
* **System Metrics:** GPU memory utilization, step execution latency, FLOPS.
* **Artifacts:** Checkpoint weights, confusion matrices, Precision-Recall curves, sample prediction visualizations.

---

## 4. Model Quality Gates (Automated Promotion Criteria)

A newly trained model artifact is **never** promoted automatically without clearing strict validation quality gates:

| Pipeline Stage | Metric Requirement | Minimum Quality Gate | Action on Failure |
| :--- | :--- | :--- | :--- |
| **Detection (RT-DETR)** | Validation mAP@50 | $\ge 95.0\%$ | Block promotion, retain current production weights |
| **Detection (RT-DETR)** | Validation mAP@50:95 | $\ge 80.0\%$ | Block promotion |
| **Verification (ResNet50)** | Validation Equal Error Rate (EER) | $\le 8.0\%$ | Block promotion |
| **Verification (ResNet50)** | ROC-AUC | $\ge 94.0\%$ | Block promotion |

---

## 5. Zero-Downtime Hot-Reload & Rollback Mechanism

1. **Atomic Weight Swapping:** Worker processes maintain a local file watcher or periodic S3 polling loop against `s3://sig-model-registry/models/latest.json`.
2. **In-Flight Task Protection:** When a new semantic version tag is detected:
   - The worker finishes processing any currently active document job from SQS.
   - Loads the new PyTorch weights into GPU memory in a background thread.
   - Re-points the active inference pointer to the new model instance.
3. **Automated Rollback:** If the worker detects more than 3 consecutive inference errors on the new weights, it automatically re-loads the previous verified checkpoint from S3 and posts an error alert.
