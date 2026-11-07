# Remote GPU Training Setup: Google Colab & Kaggle

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Target Environments:** Google Colab (T4 / V100 / A100) & Kaggle GPU (2x T4 / P100)  
**Experiment Tracking:** Comet ML & MLflow  

---

## 1. Why Remote GPU Training?
The local development environment is CPU-based. Training object detection models (RT-DETR, YOLOX-S) and metric learning backbones (ResNet50, ViT) requires dedicated tensor cores for gradient updates, mixed precision (FP16/BF16), and fast data loader throughput.

---

## 2. Google Colab Setup (Primary GPU Environment)

### Step 1: Open the Pre-built Notebook
Open [`training/remote_gpu_training_notebook.ipynb`](file:///e:/sign_detect-verify/training/remote_gpu_training_notebook.ipynb) in Google Colab:
* Go to [colab.research.google.com](https://colab.research.google.com)
* Select the **GitHub** tab and paste: `https://github.com/9m2usman/sign-detect-verify`
* Open `training/remote_gpu_training_notebook.ipynb`

### Step 2: Configure Hardware Accelerator
* Click **Runtime** $\rightarrow$ **Change runtime type**.
* Hardware accelerator: **T4 GPU** (or A100 if Colab Pro).

### Step 3: Mount Google Drive for Persistent Checkpoint Storage
```python
from google.colab import drive
drive.mount('/content/drive')
```
*Checkpoints are continuously saved to `/content/drive/MyDrive/signature_checkpoints/` so no progress is lost if the session times out.*

### Step 4: Configure Experiment Tracking Secret
```python
import os
os.environ["COMET_API_KEY"] = "your_key_here"  # Or use Colab Secrets (Lock icon on left panel)
os.environ["COMET_PROJECT_NAME"] = "signature-detection-verify"
```

### Step 5: Execute Training
Run the respective cells for:
* **RT-DETR Detection:** `!python training/detection/train_rtdetr.py`
* **YOLOX-S Detection:** `!python training/detection/train_yolox.py`
* **ResNet50 Metric Verification:** `!python training/verification/train_resnet50.py`
* **ViT Verification:** `!python training/verification/train_vit.py`

---

## 3. Kaggle GPU Setup (Secondary / Fallback Environment)

1. Create a new Notebook on [kaggle.com/code](https://www.kaggle.com/code).
2. Under **Notebook settings** (right sidebar):
   * Accelerator: **GPU T4 x 2** (30 free hours/week).
   * Internet: **On**.
3. Clone repository and run training:
   ```bash
   !git clone https://github.com/9m2usman/sign-detect-verify.git
   %cd sign-detect-verify
   !pip install -r requirements.txt
   !python training/detection/train_rtdetr.py --epochs 30 --batch-size 16
   ```
4. Output checkpoints are automatically saved to `/kaggle/working/checkpoints/` and downloadable from the Kaggle dashboard.
