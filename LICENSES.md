# Third-Party Licenses and Legal Notices

This project, **Signature Detection, Verification & Automated MLOps Platform**, is distributed under the **Apache License 2.0**.

All third-party libraries, model architectures, frameworks, and datasets utilized in this system have been audited to ensure strict compliance with commercial enterprise standards. Under no circumstances does this repository include or link against AGPL-3.0 libraries (such as Ultralytics).

---

## Direct Dependencies and Framework Licenses

| Package / Component | Author / Organization | License Type | Official Source |
| :--- | :--- | :--- | :--- |
| **PyTorch** | Meta Platforms / Linux Foundation | **BSD-3-Clause** | https://github.com/pytorch/pytorch |
| **Torchvision** | PyTorch Contributors | **BSD-3-Clause** | https://github.com/pytorch/vision |
| **OpenMMLab (MMDetection / MMYOLO)** | OpenMMLab Contributors | **Apache License 2.0** | https://github.com/open-mmlab/mmdetection |
| **YOLOX** | Megvii Technology | **Apache License 2.0** | https://github.com/Megvii-BaseDetection/YOLOX |
| **FastAPI** | Sebastián Ramírez | **MIT License** | https://github.com/tiangolo/fastapi |
| **Uvicorn** | Encode OSS | **BSD-3-Clause** | https://github.com/encode/uvicorn |
| **Pydantic** | Samuel Colvin / Pydantic Services | **MIT License** | https://github.com/pydantic/pydantic |
| **SQLAlchemy** | Mike Bayer / SQLAlchemy Authors | **MIT License** | https://github.com/sqlalchemy/sqlalchemy |
| **Celery** | Ask Solem / Celery Project | **BSD-3-Clause** | https://github.com/celery/celery |
| **RabbitMQ** | Broadcom / VMware | **Mozilla Public License 2.0** | https://www.rabbitmq.com/ |
| **psycopg (psycopg3)** | Daniele Varrazzo | **LGPL-3.0 / BSD exception** | https://www.psycopg.org/ |
| **MLflow** | Databricks / Linux Foundation | **Apache License 2.0** | https://github.com/mlflow/mlflow |
| **Pillow (PIL)** | Jeffrey A. Clark / Pillow Contributors | **HPND** | https://github.com/python-pillow/Pillow |
| **NumPy** | NumPy Developers | **BSD-3-Clause** | https://github.com/numpy/numpy |
| **OpenCV Python (opencv-python-headless)** | OpenCV team | **Apache License 2.0** | https://github.com/opencv/opencv |
| **PyTest** | Holger Krekel / Pytest Development Team | **MIT License** | https://github.com/pytest-dev/pytest |
| **Requests** | Kenneth Reitz / PSF | **Apache License 2.0** | https://github.com/psf/requests |

---

## Dataset Licensing Notice

### 1. Signature Detection Dataset (`merged_signature_dataset`)
- **Source:** Hugging Face Hub / Public Roboflow Signature Detection repository.
- **Format:** YOLO v1.1 bounding box format (`class x_center y_center width height`).
- **Usage:** Academic and non-commercial machine learning research and benchmarking.
- **Redistribution Policy:** Stored strictly locally in developer environment and Google Colab runtime; excluded from public Git repositories via `.gitignore`.

### 2. Signature Verification Benchmarks (CEDAR / BHSig260)
- **CEDAR (Center of Excellence for Document Analysis and Recognition):** Publicly available research dataset of genuine and forged signatures.
- **BHSig260 (Bengali & Hindi Signature Dataset):** Publicly available research dataset for biometric verification.
- **Usage:** Benchmarking writer-independent metric learning.

---

## License Prohibitions and Safeguards

The following license families are strictly barred from production dependencies in this project:
- **GNU Affero General Public License (AGPL-3.0)** — strictly prohibited.
- **GPL-3.0 / GPL-2.0** — strictly prohibited in backend API and production container distribution.

Verification that no AGPL libraries are installed is performed automatically as part of the continuous integration pipeline.
