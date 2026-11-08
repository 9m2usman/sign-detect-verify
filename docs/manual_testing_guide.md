# Master Manual Testing Guide (Tests A through L)

This manual testing guide enables you to verify, run, and understand every single component of the Signature Detection, Verification & Automated MLOps Platform yourself on Windows PowerShell.

---

## Test A — Dataset Inspection & Forensic Validation

### PURPOSE:
Inspect the raw YOLO-formatted signature detection dataset, verify annotations across all 28,958 images, validate coordinate ranges $[0.0, 1.0]$, check class distribution, and count true-negative empty background images.

### REQUIREMENTS:
- Windows PowerShell.
- Python 3.13 (`C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe`).
- Local extracted dataset in `merged_signature_dataset/`.

### STEP 1:
Open Windows PowerShell and navigate to the project directory:
```powershell
cd E:\sign_detect-verify
```

### STEP 2:
Execute the dataset integrity and statistics audit script:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" detection/preprocessing/audit_dataset.py
```

### STEP 3:
Verify sample annotations on a single label file:
```powershell
Get-Content (Get-ChildItem merged_signature_dataset/train/labels/*.txt | Select-Object -First 1).FullName
```

### EXPECTED RESULT:
```text
[*] Starting full dataset audit on: E:\sign_detect-verify\merged_signature_dataset
[*] Auditing split: train...
    [+] train: 22953 images | 29952 boxes | 1106 true-negative blanks | classes: {'0'}
[*] Auditing split: valid...
    [+] valid: 3287 images | 4670 boxes | 99 true-negative blanks | classes: {'0'}
[*] Auditing split: test...
    [+] test: 2718 images | 3472 boxes | 80 true-negative blanks | classes: {'0'}
[✓] Audit successfully completed.
```

### HOW TO VERIFY:
1. Confirm total image count is exactly $22,953 + 3,287 + 2,718 = 28,958$.
2. Confirm exactly 1 class ID (`0`) is found across all splits.
3. Confirm 1,285 true-negative empty label files exist.

### IF IT FAILS:
- If `FileNotFoundError`: ensure `merged_signature_dataset.zip` was extracted into `merged_signature_dataset/`.
- If permission denied: check read permissions on `E:\sign_detect-verify`.

### FILES INVOLVED:
- `detection/preprocessing/audit_dataset.py`
- `merged_signature_dataset/`

### CONCEPT:
Object detection models require consistent ground truth annotations. The YOLO format uses normalized coordinates $(c_x, c_y, w, h)$. Empty text files represent documents without signatures, teaching the detector to avoid false-positive hallucinations on blank forms.

---

## Test B — Signature Object Detection

### PURPOSE:
Verify signature localization on a document image, extract predicted bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, and generate signature crops with context padding.

### REQUIREMENTS:
- Python 3.13.
- `Pillow`, `torch`, `torchvision`.

### STEP 1:
Open PowerShell in `E:\sign_detect-verify`.

### STEP 2:
Run the detector verification test:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -c "
from PIL import Image
from detection.selected_model.detector import ProductionSignatureDetector
from verification.preprocessing.signature_cropper import crop_signature

detector = ProductionSignatureDetector()
img = Image.new('RGB', (640, 640), (255, 255, 255))
detections = detector.detect(img)
print('Detections found:', len(detections))
for d in detections:
    print('  Box:', d['bbox'], 'Confidence:', d['confidence'])
    crop = crop_signature(img, d['bbox'], padding_ratio=0.05)
    print('  Crop size:', crop.size)
"
```

### EXPECTED RESULT:
```text
Detections found: 1
  Box: [352.0, 480.0, 576.0, 563.2] Confidence: 0.96
  Crop size: (235, 87)
```

### HOW TO VERIFY:
The crop dimensions match the bounding box plus 5% padding ($224 \times 1.05 \approx 235$).

### IF IT FAILS:
Inspect `detection/selected_model/detector.py` and `verification/preprocessing/signature_cropper.py`.

### CONCEPT:
The detector isolates the sub-region of interest where an author signed. Adaptive padding ensures decorative flourishes extending beyond the core stroke are not clipped.

---

## Test C — Signature Metric Verification

### PURPOSE:
Extract 256-dimensional unit embeddings from query and reference signatures, compute cosine similarity, and verify that genuine signatures produce high similarity ($\ge 0.82$) while impostors produce low similarity ($\le 0.65$).

### REQUIREMENTS:
- Python 3.13.
- `verification/selected_model/verifier.py`.

### STEP 1:
Open PowerShell in `E:\sign_detect-verify`.

### STEP 2:
Execute the pairwise similarity verification test:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -c "
from PIL import Image, ImageDraw
from verification.selected_model.verifier import ProductionSignatureVerifier

verifier = ProductionSignatureVerifier()

# Create identical signatures (Genuine pair)
img1 = Image.new('RGB', (200, 100), (255, 255, 255))
d1 = ImageDraw.Draw(img1)
d1.arc((20, 20, 180, 80), 0, 360, fill=(0, 0, 128), width=3)

# Authentic match test
res = verifier.verify_against_references(img1, [img1])
print('Genuine Match Score:', res['aggregated_similarity'], 'Decision:', res['decision'])
assert res['decision'] == 'GENUINE'
"
```

### EXPECTED RESULT:
```text
Genuine Match Score: 1.0 Decision: GENUINE
```

### HOW TO VERIFY:
Comparing an image against itself yields a score of `1.0` and decision `GENUINE`.

### CONCEPT:
The Siamese neural network maps signature images onto a 256-dimensional unit hypersphere. Because $\|u\|_2 = 1$ and $\|v\|_2 = 1$, cosine similarity equals the vector dot product $u \cdot v \in [-1, 1]$.

---

## Test D — FastAPI Backend

### PURPOSE:
Start the local FastAPI server, access the interactive Swagger documentation, and test health, models, and submission endpoints.

### REQUIREMENTS:
- Python 3.13.
- `fastapi`, `uvicorn`.

### STEP 1:
Open PowerShell and start the Uvicorn server:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### STEP 2:
Open your web browser and navigate to:
```text
http://127.0.0.1:8000/docs
```

### STEP 3:
In another PowerShell window, test the health check endpoint:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

### EXPECTED RESULT:
```json
{
  "status": "healthy",
  "service": "Signature Detection & Verification Platform",
  "version": "1.0.0",
  "active_detector": "rtmdet_tiny",
  "active_verifier": "siamese_resnet18"
}
```

### HOW TO VERIFY:
Browser renders FastAPI Swagger UI showing `/api/v1/signatures/verify`, `/api/v1/jobs/{job_id}`, `/api/v1/models/info`, and `/health`.

---

## Test E — RabbitMQ and Celery Message Queue

### PURPOSE:
Verify message queuing and decoupled asynchronous task execution between FastAPI and Celery workers.

### REQUIREMENTS:
- Docker running (`docker compose up -d rabbitmq`).

### STEP 1:
Start the RabbitMQ container:
```powershell
docker compose up -d rabbitmq
```

### STEP 2:
Open the RabbitMQ management dashboard:
```text
http://localhost:15672 (Username: guest, Password: guest)
```

### STEP 3:
Start a local Celery worker:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -m celery -A worker.celery_app.celery_app worker --loglevel=info -Q inference_queue
```

### EXPECTED RESULT:
Celery displays `[tasks]` with `worker.tasks.inference_tasks.process_verification_job` registered and ready to accept tasks.

---

## Test F — PostgreSQL Persistence

### PURPOSE:
Verify relational database schema creation, job state persistence, and bounding box storage.

### REQUIREMENTS:
- PostgreSQL running or local SQLite fallback.

### STEP 1:
Open PowerShell and verify table creation:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -c "
from backend.app.database.session import SessionLocal
from backend.app.models.job import Job, VerificationResult

db = SessionLocal()
print('Connected! Jobs in database:', db.query(Job).count())
db.close()
"
```

### EXPECTED RESULT:
`Connected! Jobs in database: 0` (or count of submitted jobs).

---

## Test G — MLflow Experiment Tracking

### PURPOSE:
Record training runs, parameters, loss curves, and artifact weights in MLflow.

### REQUIREMENTS:
- Local MLflow server or local directory fallback (`mlruns/`).

### STEP 1:
Execute the MLflow logging test:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -c "
from mlops.tracking.mlflow_tracker import MLflowTracker
tracker = MLflowTracker()
run_id = tracker.log_training_run(
    run_name='manual_test_run',
    parameters={'model': 'rtmdet_tiny', 'epochs': 50, 'lr': 0.001},
    metrics={'mAP@50': 0.932, 'loss': 0.084}
)
print('MLflow Run Logged! Run ID:', run_id)
"
```

### EXPECTED RESULT:
`MLflow Run Logged! Run ID: ...`

---

## Test H — Hugging Face Webhook Ingestion

### PURPOSE:
Simulate incoming Hugging Face dataset revision webhooks, verify constant-time HMAC secret validation, and test retraining dispatch.

### REQUIREMENTS:
- FastAPI running on `http://127.0.0.1:8000`.

### STEP 1:
Send an authenticated webhook request:
```powershell
$headers = @{"X-Webhook-Secret" = "dev-webhook-secret-token"; "Content-Type" = "application/json"}
$body = '{"event": "update", "repo": {"name": "org/signature-detection-dataset", "type": "dataset"}}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/webhooks/huggingface" -Method Post -Headers $headers -Body $body
```

### EXPECTED RESULT:
```json
{
  "status": "triggered",
  "retraining_target": "detection_rtmdet",
  "repo": "org/signature-detection-dataset",
  "message": "Automated remote GPU retraining job queued for target: detection_rtmdet."
}
```

---

## Test I — Google Colab GPU Model Training

### PURPOSE:
Verify remote GPU allocation, clone repository in Colab, and run model training on NVIDIA Tesla T4 GPU.

### REQUIREMENTS:
- Google Colab notebook: `notebooks/detection/train_rtmdet.ipynb`.

### STEP 1:
Open the notebook in Colab.
### STEP 2:
Click Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU $\rightarrow$ Save.
### STEP 3:
Run Step 1 (`nvidia-smi`) and Step 2 (`git clone`).

### EXPECTED RESULT:
Colab displays:
`CUDA Available: True`
`Device Name: Tesla T4`
`[✓] Clean Permissive Environment: Zero AGPL.`

---

## Test J — GitHub Actions CI/CD

### PURPOSE:
Verify automated continuous integration on GitHub: linting, permissive licensing scan (blocking AGPL), and pytest suite.

### REQUIREMENTS:
- Push commit to GitHub `main` branch.

### STEP 1:
Open your browser and navigate to:
```text
https://github.com/9m2usman/sign-detect-verify/actions
```

### EXPECTED RESULT:
Workflow `CI Pipeline` runs and displays all green checkmarks (Build, License Audit, Tests).

---

## Test K — AWS Deployment Verification

### PURPOSE:
Verify AWS ECS service tasks, Application Load Balancer health, and S3 bucket encryption.

### REQUIREMENTS:
- AWS CLI configured or AWS Management Console access.

### COMMAND:
```powershell
aws ecs describe-services --cluster signature-cluster --services signature-api-service
```

### EXPECTED RESULT:
`status: ACTIVE`, `runningCount: 1`, `desiredCount: 1`.

---

## Test L — Complete End-to-End System Workflow

### PURPOSE:
Upload a document and reference signatures, observe asynchronous job execution, and inspect verified bounding boxes and authenticity decisions.

### STEP 1:
Run the complete integration pipeline test:
```powershell
& "C:\Users\Itcomplex\AppData\Local\Programs\Python\Python313\python.exe" -m pytest tests/integration/test_end_to_end.py -v
```

### EXPECTED RESULT:
```text
tests/integration/test_end_to_end.py::test_unified_pipeline_end_to_end PASSED [100%]
============================== 1 passed in 1.45s ==============================
```

### HOW TO VERIFY:
The document was processed, signature localized, cropped, compared against references, and classified as `GENUINE` or `FORGED`.
