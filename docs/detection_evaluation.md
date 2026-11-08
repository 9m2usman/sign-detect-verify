# Signature Detection Evaluation Methodology

## 1. Metrics Defined

### 1.1 Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
A predicted bounding box is considered a True Positive if $\text{IoU} \ge 0.5$ against a ground truth signature annotation.

### 1.2 Mean Average Precision (mAP)
- **mAP@0.5**: Average precision computed across all test document images with an IoU threshold of 0.5.
- **mAP@0.5:0.95**: Standard COCO metric averaging precision over 10 IoU thresholds from 0.50 to 0.95 in 0.05 increments.

## 2. Evaluation Protocol
- **Evaluation Dataset**: Disjoint holdout set of 500 diverse document layouts (invoices, legal contracts, checks, forms).
- **Inference Configuration**: Non-Maximum Suppression (NMS) threshold of 0.3, score threshold of 0.5.
- **Precision-Recall Curve**: Evaluated per document category to verify robust localization across varying background clutter.
