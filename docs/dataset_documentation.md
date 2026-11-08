# Signature Dataset Forensic Audit & Documentation

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Datasets Audited:** 
1. **Detection Dataset:** `merged_signature_dataset/` (YOLO format, 28,958 images)
2. **Verification Benchmark:** CEDAR Offline Signature Dataset & BHSig260

---

## 1. Detection Dataset Audit (`merged_signature_dataset`)

### 1.1 Overview & Architecture Suitability
The detection dataset is tailored for **Signature Object Detection (Stage 1)**. Documents represent standard letter/A4 scanned receipts, contracts, forms, and documents containing zero, one, or multiple signatures.

- **Primary Archive:** `merged_signature_dataset.zip` (3,501,396,355 bytes / 3.26 GB)
- **Extracted Root:** `merged_signature_dataset/` (57,918 files across train/valid/test splits)
- **Annotation Format:** Darknet / YOLO standard normalized coordinate system:
  $$\text{Format: } \langle\text{class\_id}\rangle \quad \langle x_{\text{center}}\rangle \quad \langle y_{\text{center}}\rangle \quad \langle\text{width}\rangle \quad \langle\text{height}\rangle$$
  All coordinate values are normalized to $[0.0, 1.0]$ relative to image dimensions.

### 1.2 Split & Annotation Statistics (Programmatically Verified)

| Split | Image Files | Label Files | Images with Annotations | True-Negative Backgrounds | Unique Classes | Total Bounding Boxes | Standard Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 22,953 | 22,953 | 21,847 | 1,106 | `0` (signature) | 29,952 | 640x640 RGB JPEG |
| **Valid** | 3,287 | 3,287 | 3,188 | 99 | `0` (signature) | 4,670 | 640x640 RGB JPEG |
| **Test** | 2,718 | 2,718 | 2,638 | 80 | `0` (signature) | 3,472 | 640x640 RGB JPEG |
| **Total** | **28,958** | **28,958** | **27,673** | **1,285** | `0` (signature) | **38,094** | **640x640 RGB JPEG** |

### 1.3 Key Forensic Findings
1. **Class Distribution:** Exactly 1 class (`class_id = 0`, denoting "signature") is present across all splits.
2. **Hard Negative Mining:** Exactly **1,285 images** contain empty label files (`0 bytes`). These are true-negative non-signature pages (forms, plain text documents, invoices without signatures) included to penalize false-positive signature hallucination.
3. **Bounding Box Density:** 
   - $72\%$ of annotated documents contain exactly 1 signature.
   - $24\%$ of annotated documents contain 2 signatures (e.g. dual party agreements).
   - $4\%$ contain $\ge 3$ signatures (e.g. multi-signatory notarized deeds).
4. **Coordinate Validity:** Zero coordinate values fell outside $[0.0, 1.0]$. Zero corrupt image files detected.

---

## 2. Verification Dataset Research & Acquisition Strategy

The detection dataset contains **zero writer identities** and cannot be used alone to train or evaluate signature biometric authenticity (Stage 2). Therefore, dedicated verification benchmarks are required:

### 2.1 Benchmark Candidate A: CEDAR Offline Signature Dataset
- **Institution:** Center of Excellence for Document Analysis and Recognition (University at Buffalo).
- **Writers:** 55 distinct signers.
- **Samples per Writer:**
  - 24 Genuine signatures.
  - 24 Skilled forgeries (produced by impostors actively attempting to mimic the target author's style).
- **Total Signatures:** $55 \times (24 + 24) = 2,640$ signatures.
- **Image Characteristics:** High-resolution grayscale / bi-tonal scans (typically 300 DPI) on clean backgrounds.
- **Suitability:** Industry gold standard for evaluating offline skilled forgery detection.

### 2.2 Benchmark Candidate B: BHSig260
- **Writers:** 260 individuals (100 Bengali and 160 Hindi writers).
- **Samples per Writer:** 24 genuine signatures and 30 skilled forgeries.
- **Total Signatures:** $260 \times 54 = 14,040$ signatures.
- **Suitability:** Cross-script biometric evaluation for non-Latin signatures.

---

## 3. Dataset Leakage Prevention Strategy

To guarantee realistic, writer-independent generalization ($W \notin \text{Train}$ during testing):

### 3.1 Writer-Disjoint Split Protocol
1. **No Overlapping Writers:** The dataset is split strictly at the **Writer ID** level, never at the image level:
   $$\text{Writers}_{\text{train}} \cap \text{Writers}_{\text{val}} = \emptyset, \quad \text{Writers}_{\text{train}} \cap \text{Writers}_{\text{test}} = \emptyset$$
   - **Training Set (70%):** 38 writers (CEDAR) / 182 writers (BHSig260).
   - **Validation Set (15%):** 8 writers (CEDAR) / 39 writers (BHSig260) — used exclusively for threshold calibration.
   - **Test Set (15%):** 9 writers (CEDAR) / 39 writers (BHSig260) — completely sealed until final model evaluation.
2. **Pairs and Triplet Generation:**
   - **Positive Pairs:** Two distinct genuine signatures from writer $W_i$ ($Y = 1$).
   - **Negative Pairs (Random Forgery):** A genuine signature from $W_i$ and a genuine signature from $W_j$ ($i \ne j$, $Y = 0$).
   - **Negative Pairs (Skilled Forgery):** A genuine signature from $W_i$ and an attempted forgery of $W_i$ produced by an impostor ($Y = 0$).

---

## 4. Preprocessing & Augmentation Pipeline

The preprocessing utilities in `verification/preprocessing/` enforce uniform input representation:
1. **Signature Cropping:** Bounding box extraction with $5\%$ adaptive margin padding to prevent clipping flourish strokes.
2. **Aspect-Ratio Letterboxing:** Preserves genuine stroke geometry without stretching or warping.
3. **Contrast Normalization:** Autocontrast enhancement separates light ink and ballpoint pen impressions from background fibers.
4. **Standard Tensor Normalization:** Converts to $\mathbb{R}^{3 \times 224 \times 224}$ tensor normalized by standard mean and variance vectors.
