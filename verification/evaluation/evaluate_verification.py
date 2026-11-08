"""
Signature Verification Metric Evaluator & Threshold Calibrator
Computes False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER),
Area Under ROC Curve (ROC-AUC), and determines calibrated operational thresholds for:
- GENUINE
- FORGED
- REVIEW (Uncertainty zone)
"""

import numpy as np
from typing import Dict, Any, Tuple, List


def compute_verification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Evaluates verification metrics at a specific operating threshold.
    y_true: Binary array (1 = Genuine match, 0 = Forgery/Impostor)
    scores: Cosine similarity scores in range [-1.0, 1.0]
    """
    y_true = np.array(y_true).astype(int)
    scores = np.array(scores).astype(float)

    y_pred = (scores >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "far": far,
        "frr": frr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def find_equal_error_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 200
) -> Tuple[float, float]:
    """
    Finds Equal Error Rate (EER) and the corresponding optimal operating threshold where FAR ≈ FRR.
    """
    thresholds = np.linspace(-1.0, 1.0, num_thresholds)
    min_diff = float("inf")
    eer = 0.0
    best_threshold = 0.0

    for thresh in thresholds:
        metrics = compute_verification_metrics(y_true, scores, threshold=thresh)
        far = metrics["far"]
        frr = metrics["frr"]
        diff = abs(far - frr)

        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            best_threshold = thresh

    return float(eer), float(best_threshold)


def calibrate_three_tier_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_far: float = 0.02,
    target_frr: float = 0.02
) -> Dict[str, float]:
    """
    Calibrates decision boundaries for high-security applications:
    - High confidence Genuine: score >= high_threshold (FAR <= target_far)
    - High confidence Forged: score <= low_threshold (FRR <= target_frr)
    - Review required: low_threshold < score < high_threshold
    """
    thresholds = np.linspace(-1.0, 1.0, 500)
    high_thresh = 0.85
    low_thresh = 0.60

    for thresh in thresholds:
        m = compute_verification_metrics(y_true, scores, threshold=thresh)
        if m["far"] <= target_far:
            high_thresh = float(thresh)
            break

    for thresh in reversed(thresholds):
        m = compute_verification_metrics(y_true, scores, threshold=thresh)
        if m["frr"] <= target_frr:
            low_thresh = float(thresh)
            break

    if low_thresh > high_thresh:
        low_thresh, high_thresh = high_thresh, low_thresh

    return {
        "forged_threshold": low_thresh,
        "genuine_threshold": high_thresh,
        "review_margin": round(high_thresh - low_thresh, 4)
    }
