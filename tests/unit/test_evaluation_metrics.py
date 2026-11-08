"""
Unit Tests for Verification Metrics and Threshold Calibration
Tests EER, FAR, FRR calculation, and three-tier decision calibration.
"""

import numpy as np
import pytest
from verification.evaluation.evaluate_verification import (
    compute_verification_metrics,
    find_equal_error_rate,
    calibrate_three_tier_thresholds
)


def test_compute_verification_metrics():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.95, 0.85, 0.30, 0.40])
    
    metrics = compute_verification_metrics(y_true, scores, threshold=0.80)
    assert metrics["accuracy"] == 1.0
    assert metrics["far"] == 0.0
    assert metrics["frr"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_find_equal_error_rate():
    # Symmetric scores
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    
    eer, thresh = find_equal_error_rate(y_true, scores)
    assert 0.0 <= eer <= 0.2
    assert 0.4 <= thresh <= 0.6


def test_calibrate_three_tier_thresholds():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.92, 0.88, 0.85, 0.70, 0.65, 0.40, 0.30, 0.15])
    
    thresholds = calibrate_three_tier_thresholds(y_true, scores)
    assert "forged_threshold" in thresholds
    assert "genuine_threshold" in thresholds
    assert thresholds["genuine_threshold"] >= thresholds["forged_threshold"]
