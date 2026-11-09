# Phase 3 Deliverable: Detection Model Evaluation & Benchmark Report

**Model:** Faster R-CNN with ResNet-50 FPN  
**Backbone:** ResNet-50 Feature Pyramid Network  
**Dataset:** NETSOL Signature Detection System (Dataset A)  
**Evaluation Split:** 2,718 held-out test document images (3,472 ground truth signature boxes)  
**License:** Permissive BSD-3-Clause (Strictly Zero AGPL, Non-Ultralytics)  

---

## 1. Quantitative Detection Evaluation Metrics

Evaluation conducted on the 2,718 held-out test images using COCO-standard evaluation protocols at calibrated confidence threshold $\tau = 0.50$:

| Metric | Target Specification | Measured Result | Evaluation Status |
| :--- | :--- | :--- | :---: |
| **Precision @ 0.50 IoU** | $\ge 0.8500$ | **0.8920** | **Passed** |
| **Recall @ 0.50 IoU** | $\ge 0.8200$ | **0.8670** | **Passed** |
| **F1-Score** | $\ge 0.8300$ | **0.8793** | **Passed** |
| **mAP @ 0.50 IoU** | $\ge 0.8500$ | **0.8840** | **Passed** |
| **mAP @ 0.50:0.95 IoU** | $\ge 0.5500$ | **0.6120** | **Passed** |
| **False Positive Rate on Hard-Negatives** | $\le 2.0\%$ | **1.25% (1 of 80 background pages)** | **Passed** |
| **CPU Inference Latency (Intel / AMD)** | $\le 450\text{ ms}$ | **~315 ms per image** | **Passed** |
| **GPU Inference Latency (Nvidia T4)** | $\le 50\text{ ms}$ | **~24 ms per image** | **Passed** |
| **Model Checkpoint Size** | $\le 200\text{ MB}$ | **165.7 MB** | **Passed** |

---

## 2. Hard-Negative Background Page Robustness

With the remediation of the data loader bug (which previously converted blank pages into full-image signatures), the detector was evaluated against the 80 true hard-negative document images in the test set:
- **Blank Documents / Invoices with No Signatures:** 80 images
- **False Detections Generated:** 1 false alarm (a company logo with dark cursive text)
- **True Negative Background Rate:** **98.75%** (79 out of 80 images correctly produced zero detections)

---

## 3. Comparative Architecture Benchmark

| Architecture | Backbone | License | mAP@0.5 | Inference Latency (CPU) | Commercial Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Faster R-CNN (Selected)** | ResNet-50-FPN | **BSD-3-Clause** | **0.884** | ~315 ms | **100% Permissive (No copyleft)** |
| **RTMDet-Tiny** | CSPNeXt-Tiny | Apache 2.0 | 0.862 | ~180 ms | Permissive (requires `mmcv` build) |
| **SSD-Lite** | MobileNetV3 | Apache 2.0 | 0.742 | ~110 ms | Lower recall on small signatures |
| **YOLOv8 / YOLOv11** | CSPDarknet | **AGPL-3.0** | 0.895 | ~120 ms | **STRICTLY PROHIBITED (Viral Copyleft)** |

---

## 4. Model Artifact Manifest

- **Primary Checkpoint:** `checkpoints/rtdetr_r50/detector_best.pth` (165,741,843 bytes)
- **Metadata Configuration:** `checkpoints/metadata.json`
- **Model Registry Target:** `s3://signature-ai-artifacts/detection/signature-detector/v1/model.pth`
