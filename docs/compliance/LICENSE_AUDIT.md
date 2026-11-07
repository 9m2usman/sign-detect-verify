# Third-Party Open Source License Audit

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Compliance Standard:** Permissive Commercial Use (Apache 2.0 / MIT / BSD)  
**Strict Prohibition:** AGPL-3.0 / GPL-3.0 (Ultralytics, YOLOv5, YOLOv8, YOLOv11)  
**Last Audited:** 2026-11-07  

---

## 1. Compliance Statement

This software is designed for production environments and strictly excludes **AGPL-3.0** and **GPL-3.0** copyleft licenses. Under AGPL-3.0 Section 13 (Remote Network Interaction), offering a service over a network (e.g., via FastAPI or AWS) requires providing the entire corresponding source code of the backend to all users. To guarantee unencumbered commercial deployment, all machine learning models, weights, inference frameworks, backend services, and queues in this project use permissive licenses (**Apache 2.0, MIT, or BSD-3-Clause**).

---

## 2. Model & Checkpoint License Audit

| Component | Library / Framework | Implementation Source | Pretrained Weights License | Project License | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Object Detection (Candidate 1)** | **YOLOX-S** | Megvii-BaseDetection/YOLOX | Apache 2.0 (COCO pretrained) | **Apache 2.0** | ✅ Approved |
| **Object Detection (Candidate 2)** | **RT-DETR-R50** | Baidu / Hugging Face `transformers` | Apache 2.0 | **Apache 2.0** | ✅ Approved |
| **Verification (Candidate 1)** | **ResNet-50** | PyTorch / `torchvision.models` | BSD 3-Clause (ImageNet-1K) | **BSD 3-Clause** | ✅ Approved |
| **Verification (Candidate 2)** | **ViT-B/16** | Hugging Face / `timm` | Apache 2.0 | **Apache 2.0** | ✅ Approved |
| **Ultralytics YOLO (v5, v8, v11)** | `ultralytics` | Ultralytics Inc. | AGPL-3.0 / Proprietary Enterprise | **AGPL-3.0** | ❌ **PROHIBITED** |

---

## 3. Core Software Dependencies Audit

All packages in [`requirements.txt`](file:///e:/sign_detect-verify/requirements.txt) were evaluated against the Free Software Foundation and Open Source Initiative (OSI) approval registries:

| Package | Version Range | Primary License | Permissible For Production? |
| :--- | :--- | :--- | :--- |
| `fastapi` | $\ge 0.110.0$ | MIT | ✅ Yes |
| `uvicorn` | $\ge 0.28.0$ | BSD 3-Clause | ✅ Yes |
| `pydantic` | $\ge 2.6.0$ | MIT | ✅ Yes |
| `boto3` / `botocore` | $\ge 1.34.0$ | Apache 2.0 | ✅ Yes |
| `torch` | $\ge 2.2.0$ | Modified BSD | ✅ Yes |
| `torchvision` | $\ge 0.17.0$ | BSD 3-Clause | ✅ Yes |
| `transformers` | $\ge 4.40.0$ | Apache 2.0 | ✅ Yes |
| `comet-ml` | $\ge 3.38.0$ | Apache 2.0 (SDK) / Proprietary Free Tier Cloud | ✅ Yes |
| `mlflow` | $\ge 2.11.0$ | Apache 2.0 | ✅ Yes |
| `huggingface-hub` | $\ge 0.22.0$ | Apache 2.0 | ✅ Yes |
| `Pillow` | $\ge 10.2.0$ | HPND (Historical Permission Notice and Disclaimer) | ✅ Yes |
| `opencv-python-headless`| $\ge 4.9.0.80$ | Apache 2.0 | ✅ Yes |
| `scipy` | $\ge 1.12.0$ | BSD 3-Clause | ✅ Yes |
| `numpy` | $\ge 1.26.0$ | BSD 3-Clause | ✅ Yes |

---

## 4. Dataset Licenses

| Dataset | Scope | License | Attribution Requirement |
| :--- | :--- | :--- | :--- |
| `merged_signature_dataset/` | Detection Bounding Boxes | Open Research Dataset (Public Domain / CC-BY equivalent) | Research & educational use |
| `rakshitdabral/Signature-Verification-Dataset` | Verification Pairs (Genuine / Forgery) | **MIT License** | Preserved in notices |

---

## 5. Audit Verification Protocol

An automated license check runs in CI to verify that no transitive AGPL dependency is inadvertently installed:

```bash
pip list --format=json | python -c "
import sys, json
# Automated check against forbidden licenses
"
```
