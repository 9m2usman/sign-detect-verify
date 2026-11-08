# User Guide & Operations Manual

## 1. Introduction
The Signature Detection and Verification System provides automated extraction and biometric comparison of handwritten signatures from documents.

## 2. Using the System

### 2.1 Enrolling a Writer Profile
Before verifying signatures against a specific person, their genuine reference signatures should be enrolled:
1. Submit high-quality reference signatures via the enrollment endpoint or administrative dashboard.
2. The system computes and stores the 128-dimensional embedding vector for the reference samples.

### 2.2 Processing a Document
1. Upload the scanned document (contract, check, invoice).
2. The detection module locates all candidate signature zones.
3. Each candidate region is extracted as a normalized signature image.

### 2.3 Verification Decision Logic
- **Similarity Score $\ge 0.75$**: Classified as **GENUINE**.
- **Similarity Score $< 0.65$**: Classified as **FORGERY**.
- **$0.65 \le \text{Score} < 0.75$**: Flagged for **MANUAL REVIEW** (inconclusive / border case).

## 3. Best Practices for Document Capture
- Scan documents at a minimum of 200 DPI (300 DPI recommended).
- Ensure signatures are not obscured by dark background stamps or overlapping folds.
