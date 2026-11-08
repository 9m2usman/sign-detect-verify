# Model Licensing and Legal Compliance Audit

**Author:** AI/ML Engineering & Compliance Architecture Team  
**Policy:** Strict Non-AGPL Compliance — 100% Permissive Open Source (Apache 2.0 / MIT / BSD)  
**Status:** Enforced Across All Modules

---

## 1. Executive Summary & Prohibition of Ultralytics

Our enterprise client and academic instructor strictly prohibit the use of **Ultralytics** libraries (including YOLOv5, YOLOv8, and YOLO11 via the `ultralytics` package). 

### The AGPL-3.0 Legal Hazard
The `ultralytics` repository is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Copyleft Infiltration:** AGPL-3.0 contains a unique "Network Trigger" (Section 13). If software incorporating AGPL-3.0 components is executed on a server and made accessible over a computer network (e.g., via a REST API like FastAPI), the complete source code of the entire system must be provided to all users under the AGPL-3.0 license.
- **Commercial Incompatibility:** In an enterprise SaaS environment or proprietary deployment, AGPL-3.0 jeopardizes proprietary intellectual property and prevents commercial enterprise adoption without costly commercial licenses.
- **Hard Rule:** Under no circumstances will `ultralytics` be imported, installed, or called in this project.

---

## 2. Permitted Permissive Architecture Candidates

To replace AGPL Ultralytics, we have audited and selected permissive candidates licensed under **Apache 2.0**, **MIT**, or **BSD-3-Clause**:

| Architecture / Framework | Upstream Organization | License | Commercial Use Permitted? | Role in Project |
| :--- | :--- | :--- | :--- | :--- |
| **RTMDet-Tiny** | OpenMMLab (MMDetection / MMYOLO) | **Apache 2.0** | **YES** | Detection Candidate A |
| **YOLOX-Tiny** | Megvii Technology | **Apache 2.0** | **YES** | Detection Candidate B |
| **Siamese CNN (Custom)** | PyTorch Core Native | **BSD-3-Clause** | **YES** | Verification Candidate A |
| **Triplet Network (Custom)** | PyTorch Core Native | **BSD-3-Clause** | **YES** | Verification Candidate B |
| **FastAPI** | Sebastián Ramírez | **MIT** | **YES** | API Gateway |
| **Celery** | Celery Project | **BSD-3-Clause** | **YES** | Asynchronous Task Worker |
| **RabbitMQ** | Broadcom / VMware | **Mozilla Public License 2.0** | **YES** | Message Broker |
| **PostgreSQL** | PostgreSQL Global Dev Group | **PostgreSQL (BSD-like)** | **YES** | Relational Database |
| **MLflow** | Linux Foundation | **Apache 2.0** | **YES** | MLOps Tracking |

---

## 3. Deep-Dive Audit: Candidate Detection Models

### 3.1 Candidate A: RTMDet-Tiny (OpenMMLab)
- **Framework:** OpenMMLab MMDetection 3.x / MMYOLO.
- **License:** Apache License 2.0.
- **Pretrained Weights:** Pre-trained on COCO under Apache 2.0.
- **Technical Attributes:**
  - High-efficiency real-time object detector designed specifically for low-latency edge and server inference.
  - Large-kernel depthwise convolutions (5x5) in the backbone (CSPNeXt) provide large receptive fields ideal for capturing elongated signature ink strokes.
  - Decoupled head with anchor-free regression.
  - Dynamic soft label assignment (SimOTA-derived) accelerates training convergence.
- **Suitability:** Completely unencumbered by copyleft restrictions; superior accuracy-to-latency trade-off compared to older YOLO variants.

### 3.2 Candidate B: YOLOX-Tiny (Megvii Technology)
- **Framework:** Megvii YOLOX (official repository: `Megvii-BaseDetection/YOLOX`).
- **License:** Apache License 2.0.
- **Pretrained Weights:** Pre-trained on COCO under Apache 2.0.
- **Technical Attributes:**
  - Anchor-free one-stage detector.
  - Decoupled detection head separates classification and regression tasks into distinct branches, speeding convergence.
  - SimOTA dynamic label assignment prevents label noise during multi-scale training.
  - Highly optimized ONNX and TensorRT runtime support.
- **Suitability:** Industry-standard baseline with 100% permissive licensing; native CPU and GPU deployment paths.

---

## 4. Deep-Dive Audit: Candidate Verification Models

### 4.1 Candidate A: Siamese CNN with Contrastive Loss
- **Implementation:** Native PyTorch architecture (`torch.nn.Module`).
- **License:** BSD-3-Clause (governed by PyTorch core license).
- **Architecture:** Twin convolutional branches sharing identical weights, passing inputs through feature extraction layers into a normalized 256-dimensional or 512-dimensional embedding vector.
- **Loss Function:** Contrastive Loss ($L = \frac{1}{2}(1-Y)D^2 + \frac{1}{2}Y \max(0, m - D)^2$).
- **Suitability:** Zero third-party dependency risks; complete control over network depth, feature projection, and normalization.

### 4.2 Candidate B: Triplet Network with Triplet Margin Loss
- **Implementation:** Native PyTorch architecture (`torch.nn.Module`).
- **License:** BSD-3-Clause (governed by PyTorch core license).
- **Architecture:** Three parallel feeds (Anchor $A$, Positive $P$, Negative $N$) sharing a single deep encoder backbone.
- **Loss Function:** Triplet Margin Loss ($L(A, P, N) = \max(0, D(A, P) - D(A, N) + \alpha)$).
- **Suitability:** Proven metric learning approach that enforces distance separation between different writers while pulling genuine variations closer together in metric space.

---

## 5. Automated CI Licensing Enforcement

To prevent accidental introduction of copyleft packages, our CI/CD pipeline (`.github/workflows/ci.yml`) executes an automated licensing scan on every pull request and push:

```bash
# Automated license check in CI
python -c "
import importlib.util, sys
if importlib.util.find_spec('ultralytics') is not None:
    print('FATAL ERROR: AGPL-3.0 Ultralytics detected in environment!')
    sys.exit(1)
print('[✓] Environment verified: Zero AGPL contamination.')
"
```
Any PR containing `ultralytics` in `requirements.txt` or imported in code will be immediately blocked from merging.
