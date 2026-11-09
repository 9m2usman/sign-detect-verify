# Dataset Audit & Separation Report

**Document Version:** 1.0.0  
**Date:** November 2026  
**Project:** Signature Detection & Verification Platform  

---

## 1. Absolute Dataset Separation Mandate

This project strictly enforces separation between two distinct machine learning tasks and datasets:

| Characteristic | Dataset A | Dataset B |
| :--- | :--- | :--- |
| **Machine Learning Task** | **Signature Object Detection** | **Signature Verification (Biometric Metric Learning)** |
| **Purpose** | Locate signatures in document pages and return bounding boxes | Compare signature crops to reference signatures and score similarity |
| **Origin Repository** | `netsol-signature-detection/Signature-Detection-System` | Academic / Public Verification Dataset (e.g., CEDAR, BHSig260) |
| **Contains Writer IDs?** | **NO** (Zero writer identities) | **YES** (Signatures grouped by writer) |
| **Contains Genuine/Forged Labels?** | **NO** (Only bounding boxes) | **YES** (Genuine vs. Skilled/Simple Forgery labels) |
| **Separation Rule** | Never use for verification training | Never mix with detection documents |

---

## 2. Dataset A (Detection) Audit

### 2.1 File Location and Integrity
- **Local Directory:** `E:\sign_detect-verify\merged_signature_dataset\`
- **Archive Source:** `E:\sign_detect-verify\merged_signature_dataset.zip` (3,501,396,355 bytes / 3.26 GiB)
- **Hugging Face Hub ID:** `netsol-signature-detection/Signature-Detection-System`
- **Total Image Files:** 28,958
- **Total Label Files:** 28,958
- **Format:** Normalized YOLO format (`class x_center y_center width height`), with `class = 0 (signature)`.

### 2.2 Split Distribution and Metrics

```text
merged_signature_dataset/
├── train/          # 22,953 images, 22,953 txt files, 29,952 signature boxes (1,106 background negatives)
├── valid/          #  3,287 images,  3,287 txt files,  4,670 signature boxes (99 background negatives)
└── test/           #  2,718 images,  2,718 txt files,  3,472 signature boxes (80 background negatives)
```

| Split | Total Images | Images with Signatures | True-Negative Backgrounds | Total Bounding Boxes | Avg Boxes / Image |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 22,953 | 21,847 (95.18%) | 1,106 (4.82%) | 29,952 | 1.30 |
| **Validation** | 3,287 | 3,188 (96.99%) | 99 (3.01%) | 4,670 | 1.42 |
| **Test** | 2,718 | 2,638 (97.06%) | 80 (2.94%) | 3,472 | 1.28 |
| **Total** | **28,958** | **27,673 (95.56%)** | **1,285 (4.44%)** | **38,094** | **1.32** |

### 2.3 Critical Finding: Data Loader Defect in `train_rtdetr.py`
The audit discovered a logic flaw in `training/detection/train_rtdetr.py`:
```python
# Lines 78-81
if not boxes:
    boxes = [[0.0, 0.0, float(w), float(h)]]
    labels = [1]
```
The 1,285 intentionally included background non-signature pages (which are vital for training the detector not to hallucinate false positives on clean documents) are converted into bounding boxes covering the entire image with label 1 ("signature"). This corrupts training. The fix requires setting `boxes = torch.zeros((0, 4), dtype=torch.float32)` and `labels = torch.zeros((0,), dtype=torch.int64)` for unannotated images.

---

## 3. Dataset B (Verification) Research & Candidate Comparison

Dataset B is **not yet downloaded**, pending user review and approval in Phase 2. The candidate datasets have been researched as follows:

| Metric / Criterion | Candidate 1: CEDAR | Candidate 2: BHSig260 | Candidate 3: GPDS |
| :--- | :--- | :--- | :--- |
| **Signer Count** | 55 signers | 260 signers | Up to 4,000 signers |
| **Signatures / Signer** | 24 genuine, 24 forgeries | 24 genuine, 30 forgeries | 24 genuine, 30 forgeries |
| **Total Signatures** | **2,640 signatures** | **14,040 signatures** | Large scale |
| **Language / Script** | English / Latin | Bengali (100 signers) & Hindi (160 signers) | Western / Spanish |
| **Forgery Quality** | High-quality skilled forgeries | Skilled forgeries | Skilled & simple forgeries |
| **Format & Quality** | 300 DPI grayscale PNGs | Grayscale TIFF / PNG | Varying scan resolutions |
| **Licensing** | Academic research use | Academic research use | Academic research use |
| **Writer Split Strategy** | 45 train, 10 test (writer-independent) | 200 train, 60 test (writer-independent) | Writer-independent subsets |
| **Evaluation Strengths** | Clean, widely cited, standard benchmark | Diverse non-Latin scripts, larger scale | Very large signer pool |
| **Recommendation** | **Primary Recommendation** for baseline | **Secondary Recommendation** for multi-lingual extension | Complex access terms |

### Recommendation Rationale:
- **CEDAR** is the industry standard for offline signature verification research. Its balanced structure (24 genuine and 24 skilled forgeries per signer) allows clean construction of positive (genuine-genuine) and negative (genuine-forgery) pairs, and its writer-independent split guarantees zero data leakage between train and test sets.
