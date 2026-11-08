# System Requirements Specification

## 1. Functional Requirements

### 1.1 Document Ingestion & Signature Detection
- **FR-1.1**: The system shall accept standard document formats (PDF, PNG, JPEG, TIFF) up to 25 MB.
- **FR-1.2**: The system shall localize one or more signature instances within multi-page documents and return normalized bounding coordinates `[x1, y1, x2, y2]` along with detection confidence scores.
- **FR-1.3**: The detection engine shall utilize a permissively licensed model architecture (Faster R-CNN with ResNet-50-FPN) strictly excluding copyleft/AGPL codebases.
- **FR-1.4**: Target detection performance: mAP@0.5 >= 0.85 on benchmark evaluation datasets.

### 1.2 Signature Verification
- **FR-2.1**: The system shall compare an extracted or query signature against a genuine reference signature from a declared writer profile.
- **FR-2.2**: The verification engine shall output a normalized similarity score $[0.0, 1.0]$ based on cosine distance between L2-normalized deep embedding vectors.
- **FR-2.3**: The verification pipeline shall evaluate a classification decision (Genuine vs. Forgery) against a calibrated threshold that balances False Acceptance Rate (FAR) and False Rejection Rate (FRR).
- **FR-2.4**: Target verification performance: Equal Error Rate (EER) <= 12% across multi-script signature benchmarks (CEDAR, BHSig).

### 1.3 Asynchronous Job Queue & Storage
- **FR-3.1**: Heavy inference operations shall be processed asynchronously via Celery workers orchestrated through RabbitMQ.
- **FR-3.2**: Job status (QUEUED, PROCESSING, COMPLETED, FAILED) and intermediate detection/verification results shall be stored in PostgreSQL.

## 2. Non-Functional Requirements

### 2.1 Licensing and Legal Compliance
- **NFR-1.1**: All source code, dependencies, pretrained model weights, and datasets shall adhere strictly to permissive licensing (Apache 2.0, MIT, BSD).
- **NFR-1.2**: Ultralytics YOLO models (GPL-3.0 / AGPL-3.0) are strictly prohibited.

### 2.2 Performance & Scalability
- **NFR-2.1**: API response latency for job dispatch shall be < 150 ms.
- **NFR-2.2**: Celery worker inference latency per document page shall be < 1.5 seconds on standard CPU.

### 2.3 Reliability & Fault Tolerance
- **NFR-3.1**: Unhandled exceptions in worker tasks shall be captured with retry semantics and failure reason logged to PostgreSQL.
