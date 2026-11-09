# Phase 2 Deliverable: Machine Learning Evaluation Plan

**Project:** Signature Detection & Verification Platform  
**Scope:** Quantitative Metrics, Threshold Calibration, and Baseline Benchmarking  

---

## 1. Stage 1: Detection Model Evaluation Protocol

### 1.1 Metrics & Mathematical Definitions

1. **Intersection over Union (IoU):**
   $$\text{IoU} = \frac{\text{Area}(\text{Box}_{\text{pred}} \cap \text{Box}_{\text{ground\_truth}})}{\text{Area}(\text{Box}_{\text{pred}} \cup \text{Box}_{\text{ground\_truth}})}$$
   A predicted bounding box is counted as a True Positive (TP) if $\text{IoU} \ge \text{threshold}$ (typically 0.50) and False Positive (FP) otherwise.

2. **Precision and Recall:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

3. **Mean Average Precision (mAP@0.5 and mAP@0.5:0.95):**
   $$\text{AP} = \int_{0}^{1} p(r) \, dr$$
   - $\text{mAP@0.5}$: Average Precision evaluated at fixed $\text{IoU} = 0.50$.
   - $\text{mAP@0.5:0.95}$: Average Precision averaged across 10 IoU thresholds from $0.50$ to $0.95$ in steps of $0.05$.

4. **Inference Latency & Computational Footprint:**
   - Inference time per 640×640 image on CPU (target: $< 400\text{ ms}$).
   - Model parameter count (Faster R-CNN ResNet-50: $\approx 41.5\text{M}$ parameters).
   - Checkpoint file size on disk ($\approx 165.7\text{ MB}$).

---

## 2. Stage 2: Verification Model Evaluation Protocol

### 2.1 Metrics & Mathematical Definitions

1. **False Acceptance Rate (FAR):**
   The proportion of forged signatures that are incorrectly accepted as genuine:
   $$\text{FAR}(\tau) = \frac{\text{Number of forgeries with score } \ge \tau}{\text{Total number of forged comparisons}}$$

2. **False Rejection Rate (FRR):**
   The proportion of genuine signatures that are incorrectly rejected as forgeries:
   $$\text{FRR}(\tau) = \frac{\text{Number of genuine signatures with score } < \tau}{\text{Total number of genuine comparisons}}$$

3. **Equal Error Rate (EER):**
   The operating point where the False Acceptance Rate equals the False Rejection Rate:
   $$\text{EER} = \text{FAR}(\tau^*) = \text{FRR}(\tau^*) \quad \text{where } \tau^* \text{ is the EER threshold}$$
   A lower EER represents superior discrimination between genuine and skilled forgery signatures.

4. **Receiver Operating Characteristic (ROC) & Area Under Curve (ROC-AUC):**
   The ROC curve plots $\text{True Accept Rate} = 1 - \text{FRR}$ against $\text{FAR}$ across all possible threshold values $\tau \in [0.0, 1.0]$. The area under the ROC curve (ROC-AUC) quantifies overall discriminative capability (target: $\ge 0.95$).

5. **Structural Similarity Index (SSIM Baseline):**
   Traditional non-learned computer vision baseline comparing luminance ($l$), contrast ($c$), and structure ($s$):
   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   The SSIM baseline will be benchmarked alongside the Siamese ResNet model to quantify the performance gain achieved by deep metric learning.

---

## 3. Strict Threshold Calibration Rules

> [!CAUTION]
> **Zero Data Leakage Rule for Threshold Calibration:**
> The verification decision threshold $\tau$ must be calibrated exclusively on the **Validation Split** (Writers 41 to 45 in CEDAR). Under no circumstances may the threshold be tuned on the final Test Set (Writers 46 to 55). Tuning on test data produces over-optimistic evaluation metrics that fail in real-world deployment.

### Threshold Selection Methodology:
1. Sweep $\tau$ from $0.00$ to $1.00$ in increments of $0.005$ on validation genuine-genuine and genuine-forgery pairs.
2. Identify $\tau_{\text{EER}}$ where $|\text{FAR}(\tau) - \text{FRR}(\tau)|$ is minimized.
3. Compute business-calibrated thresholds:
   - **Balanced Operating Threshold ($\tau_{\text{balanced}}$):** Set at $\tau_{\text{EER}}$.
   - **High-Security Operating Threshold ($\tau_{\text{secure}}$):** Set where $\text{FAR} \le 1.0\%$.
4. Lock in $\tau$ and evaluate once on the held-out test split.
