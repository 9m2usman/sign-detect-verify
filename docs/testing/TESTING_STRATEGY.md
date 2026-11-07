# Testing Strategy, Quality Assurance & Benchmark Suite

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Testing Framework:** `pytest` + `fastapi.testclient`  
**Execution Host:** Windows 11 / CI GitHub Actions  

---

## 1. Testing Philosophy & Guardrails

1. **Evidence-Based QA:** Every metric, calculation, and bounding box transformation is tested against known geometric truths.
2. **Zero Fake Mocks in Domain Logic:** Math operations (IoU, coordinate transformation, L2-normalization, cosine similarity) run against pure Python and PyTorch implementations without mocked math.
3. **Mocked Cloud Boundaries:** AWS S3 and SQS interactions are isolated using `unittest.mock` during unit tests and verified with LocalStack during integration tests.

---

## 2. Test Suite Architecture

```
tests/
├── test_ml_pipeline.py     # Unit tests for BoundingBox, IoU, Crops, Normalization, Dataset
└── test_api.py             # Integration tests for FastAPI endpoints, S3 staging, SQS dispatch
```

### 2.1 Coverage Matrix

| Test Function | Target Component | Verification Objective | Result |
| :--- | :--- | :--- | :--- |
| `test_bounding_box_conversion` | `ml.dataset.BoundingBox` | Validates normalized YOLO $[cx, cy, w, h] \to [xmin, ymin, xmax, ymax]$ | ✅ PASSED |
| `test_compute_iou` | `ml.utils.compute_iou` | Tests complete overlap ($\text{IoU} = 1.0$) and disjoint boxes ($\text{IoU} = 0.0$) | ✅ PASSED |
| `test_extract_signature_crops` | `ml.utils.extract_signature_crops` | Tests padding ratio expansion without border violation | ✅ PASSED |
| `test_preprocess_signature_normalization` | `ml.utils.preprocess_signature_for_verification` | Verifies aspect-ratio letterboxing to $224 \times 224 \times 3$ | ✅ PASSED |
| `test_dataset_parser_local_split` | `ml.dataset.YOLODatasetParser` | Validates parsing of real 3,287 validation images in local dataset | ✅ PASSED |
| `test_health_endpoint` | `api.main.health` | Verifies ALB target health check probe returns 200 OK | ✅ PASSED |
| `test_submit_signature_verification_invalid_mime` | `api.main.submit_signature_verification` | Asserts non-image inputs are rejected with HTTP 400 | ✅ PASSED |
| `test_submit_signature_verification_success` | `api.main.submit_signature_verification` | Validates S3 upload + SQS dispatch returns HTTP 202 | ✅ PASSED |
| `test_get_job_status_completed` | `api.main.get_job_status` | Tests S3 result parsing and JSON serialization | ✅ PASSED |

---

## 3. How to Run Tests Locally
```powershell
# Run with Python test runner
python -m pytest tests/ -v
```
