# MLOps & Experiment Tracking Pipeline

## 1. Tooling & Architecture
- **Tracking Server**: MLflow Server backed by SQLite/PostgreSQL metadata store and S3/local artifact repository.
- **Model Registry**: Versions and stages (`Staging`, `Production`, `Archived`) tracked programmatically.
- **Reproducibility**: Environment parameters, random seeds, Git commit hashes, and dataset versions recorded for every run.

## 2. Tracking Parameters & Metrics

### Detection Runs
- **Parameters**: `learning_rate`, `batch_size`, `num_epochs`, `backbone`, `anchor_sizes`, `nms_iou_threshold`.
- **Metrics**: `train_loss`, `val_loss`, `rpn_loss_classifier`, `rpn_loss_box_reg`, `mAP@0.5`, `mAP@0.5:0.95`.

### Verification Runs
- **Parameters**: `margin`, `embedding_dim`, `optimizer`, `loss_type` (contrastive/triplet).
- **Metrics**: `contrastive_loss`, `eer_validation`, `roc_auc`, `far_optimal`, `frr_optimal`.

## 3. Automation & Webhooks
- Model promotion triggers automated regression tests against standard benchmark suites before transitioning to `Production`.
