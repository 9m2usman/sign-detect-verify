# Phase 2 Deliverable: Experiment Tracking & Model Registry Architecture

**Tracking Platform:** MLflow  
**Backend Store:** Amazon RDS PostgreSQL (Local: SQLite `signature_dev.db`)  
**Artifact Store:** Amazon S3 `s3://signature-ai-artifacts/mlflow/` (Local: `./mlruns/`)  

---

## 1. MLflow Experiment Hierarchy

Experiments are partitioned into two isolated MLflow experiments:
1. `signature-detection-rtmdet-fasterrcnn`: Dedicated exclusively to Dataset A object detection experiments.
2. `signature-verification-siamese-metric`: Dedicated exclusively to Dataset B metric learning and Siamese verification experiments.

---

## 2. Detection Experiment Tracking Schema

### Parameters Logged
- `model_architecture`: `"fasterrcnn_resnet50_fpn"`
- `framework_version`: `"torchvision-0.29.2"`
- `dataset_name`: `"netsol-signature-detection"`
- `dataset_revision`: `"28958_imgs_commit_11c63f2"`
- `train_samples`: `22953`
- `val_samples`: `3287`
- `batch_size`: `8`
- `learning_rate`: `0.0002`
- `optimizer`: `"AdamW"`
- `scheduler`: `"CosineAnnealingLR"`
- `epochs`: `10`
- `random_seed`: `42`

### Metrics Logged (Per Epoch)
- `train/loss_total`, `train/loss_classifier`, `train/loss_box_reg`, `train/loss_rpn_box_reg`, `train/loss_objectness`
- `val/mAP_50`
- `val/mAP_50_95`
- `val/precision`
- `val/recall`
- `inference/latency_ms`

### Artifacts Stored
- `weights/model_best.pth`
- `weights/model_final.pth`
- `plots/loss_curves.png`
- `plots/pr_curve.png`
- `plots/confusion_matrix.png`
- `samples/detection_examples.jpg`
- `config/training_config.json`

---

## 3. Verification Experiment Tracking Schema

### Parameters Logged
- `model_architecture`: `"siamese_resnet50"`
- `embedding_dimension`: `256`
- `dataset_name`: `"CEDAR"`
- `margin`: `1.0`
- `learning_rate`: `0.0001`
- `split_strategy`: `"writer_independent"`
- `train_writers`: `40`
- `val_writers`: `5`
- `test_writers`: `10`

### Metrics Logged
- `train/contrastive_loss`
- `val/contrastive_loss`
- `val/eer`
- `val/roc_auc`
- `val/optimal_threshold`
- `val/far_at_threshold`
- `val/frr_at_threshold`
- `baseline/ssim_eer`
- `baseline/ssim_roc_auc`

### Artifacts Stored
- `weights/siamese_best.pth`
- `plots/roc_curve.png`
- `plots/genuine_forgery_score_distributions.png`
- `plots/far_frr_tradeoff.png`
- `samples/verification_comparisons.jpg`

---

## 4. Model Registry Automated Quality Gates

When a new model completes training, it must pass predefined quality gates before promotion to `Production`:

| Gate Condition | Detection Champion Gate | Verification Champion Gate |
| :--- | :--- | :--- |
| **Minimum Metric Baseline** | $\text{mAP@0.5} \ge 0.85$ | $\text{EER} \le 10.0\%$ |
| **Regression Check vs. Current Champion** | Candidate $\text{mAP@0.5} \ge \text{Champion} - 0.01$ | Candidate $\text{EER} \le \text{Champion} + 0.5\%$ |
| **Integrity Check** | SHA-256 checksum matches S3 record | SHA-256 checksum matches S3 record |
| **Inference Health Check** | CPU latency $\le 450\text{ ms}$, zero NaN predictions | Latency $\le 200\text{ ms}$, normalized outputs |

Models that fail any gate remain in `Staging` (or are tagged `Rejected`), and the existing `Production` champion remains actively serving inference workers without interruption.
