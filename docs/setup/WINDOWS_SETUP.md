# Local Windows 11 & Python Environment Setup Guide

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Target OS:** Windows 11 (64-bit)  
**Supported Python Versions:** Python 3.10 – 3.13  

---

## 1. Prerequisites

1. **Python 3.10+ (Recommended: Python 3.11 or 3.13):**
   Ensure Python is added to your Windows system `PATH`.
2. **Git for Windows:**
   Install Git with Windows Credential Manager.
3. **Docker Desktop for Windows (Optional for local S3/SQS simulation):**
   Required to run LocalStack for local AWS simulation.

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```powershell
git clone https://github.com/9m2usman/sign-detect-verify.git
cd sign-detect-verify
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Core Dependencies (100% Non-Ultralytics / Apache 2.0 / MIT)
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install PyTorch CPU (or CUDA if NVIDIA GPU available)
* For **CPU** (default on local Windows workstation):
  ```powershell
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
  ```
* For **NVIDIA GPU** (CUDA 12.1):
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

---

## 3. Verify Local Installation

Run the automated test runner:
```powershell
pytest tests/ -v
```
Expected output:
```
tests/test_ml_pipeline.py::test_bounding_box_conversion PASSED
tests/test_ml_pipeline.py::test_compute_iou PASSED
tests/test_ml_pipeline.py::test_extract_signature_crops PASSED
tests/test_ml_pipeline.py::test_preprocess_signature_normalization PASSED
tests/test_ml_pipeline.py::test_dataset_parser_local_split PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_submit_signature_verification_invalid_mime PASSED
tests/test_api.py::test_submit_signature_verification_success PASSED
tests/test_api.py::test_get_job_status_completed PASSED
```

---

## 4. Running the Local API & Background Worker

### Option A: Direct Local Execution (With LocalStack)
1. Start LocalStack:
   ```powershell
   docker-compose up -d localstack
   ```
2. Start FastAPI Ingestion Server:
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Open Swagger UI at: `http://localhost:8000/docs`
3. Start the Asynchronous Worker:
   ```powershell
   python -m worker.consumer
   ```
