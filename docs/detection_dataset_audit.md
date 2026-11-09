# Phase 2 Deliverable: Detection Dataset Audit (Dataset A)

**Dataset Name:** NETSOL Signature Detection System  
**Hugging Face Hub ID:** `netsol-signature-detection/Signature-Detection-System`  
**Local Archive:** `E:\sign_detect-verify\merged_signature_dataset.zip` (3.50 GB)  
**Extracted Path:** `E:\sign_detect-verify\merged_signature_dataset\`  

---

## 1. Dataset Overview & Separation Verification

Dataset A is strictly reserved for signature object detection. A forensic inspection confirms:
- **Writer Identity:** Zero writer IDs exist in the dataset labels or metadata.
- **Forgery Labels:** Zero genuine/forged classification labels exist.
- **Annotation Format:** Bounding boxes in normalized YOLO text format:
  $$\text{Format: } \langle\text{class\_id}\rangle \;\; \langle x_{\text{center}}\rangle \;\; \langle y_{\text{center}}\rangle \;\; \langle w\rangle \;\; \langle h\rangle$$
  where $\text{class\_id} = 0$ corresponds to `"signature"`, and all spatial coordinates are normalized to the range $[0.0, 1.0]$.

---

## 2. Statistical Breakdown Across Dataset Splits

| Split | Image Files | Label Files | Images with Signatures | True-Negative Backgrounds | Total Bounding Boxes | Mean Boxes / Image |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 22,953 | 22,953 | 21,847 (95.18%) | 1,106 (4.82%) | 29,952 | 1.30 |
| **Validation** | 3,287 | 3,287 | 3,188 (96.99%) | 99 (3.01%) | 4,670 | 1.42 |
| **Test** | 2,718 | 2,718 | 2,638 (97.06%) | 80 (2.94%) | 3,472 | 1.28 |
| **Total** | **28,958** | **28,958** | **27,673 (95.56%)** | **1,285 (4.44%)** | **38,094** | **1.32** |

### Image Specifications
- **Dimensions:** 640 × 640 pixels
- **Channels:** 3 (RGB)
- **File Format:** JPEG (`.jpg`)
- **Aspect Ratio:** Fixed 1:1 square canvas (documents padded or scaled)

---

## 3. Data Integrity & Negative Sample Verification

### True Hard-Negative Background Pages
The presence of **1,285 non-signature documents** (1,106 in train, 99 in validation, 80 in test) is a deliberate architectural feature of the dataset. These blank pages, contracts, invoices, and letterheads without signatures serve as negative examples to train the Region Proposal Network (RPN) to reject background noise and avoid false-positive hallucinations.

### Defect Remediation in Data Loader
The previous loader in `training/detection/train_rtdetr.py` erroneously converted empty annotations into full-page bounding boxes with label 1 ("signature"). This defect has been resolved:
- **Remediated Logic:** Images with empty label files now return empty target tensors:
  $$\text{boxes} = \text{torch.zeros}((0, 4), \text{dtype}=\text{torch.float32})$$
  $$\text{labels} = \text{torch.zeros}((0,), \text{dtype}=\text{torch.int64})$$
- **Verification:** Verified with Python 3.13 test script. Positive images yield shape `[N, 4]` and negative images yield shape `[0, 4]`.

---

## 4. Train / Validation / Test Strategy

To prevent data leakage and evaluate real-world generalization:
1. **Train Split (22,953 images):** Used for gradient updates, Region Proposal Network tuning, and ROI head optimization.
2. **Validation Split (3,287 images):** Used for checkpoint selection, loss plateau monitoring, learning rate scheduler triggers (Cosine Annealing), and hyperparameter tuning.
3. **Test Split (2,718 images):** Held out completely until training is finalized. Used to calculate unbiased final evaluation metrics:
   - $\text{mAP@0.5}$
   - $\text{mAP@0.5:0.95}$
   - Precision and Recall at calibrated confidence threshold $\tau = 0.50$
