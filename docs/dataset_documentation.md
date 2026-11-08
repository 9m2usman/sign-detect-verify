# Dataset Documentation & Governance

## 1. Overview & Master Archive
The core training and evaluation data consists of the consolidated offline signature dataset:
- **Zip Archive**: `merged_signature_dataset.zip` (3.26 GB, 3,501,396,355 bytes)
- **Extracted Root**: `merged_signature_dataset/` (3.47 GB, 57,918 files)

Both master assets are strictly preserved and guarded against modifications or accidental deletion.

## 2. Benchmark Datasets Included

### 2.1 CEDAR (Center of Excellence for Document Analysis and Recognition)
- **Signatures**: 55 signers, 24 genuine signatures and 24 skilled forgeries per signer.
- **Language/Script**: Latin / English cursive.
- **Resolution**: 300 DPI grayscale scans.
- **Primary Use**: Baseline contrastive pair evaluation for skilled forgery discrimination.

### 2.2 BHSig260
- **Bengali Sub-dataset (BHSig-B)**: 100 signers, 24 genuine and 30 skilled forgeries per signer.
- **Hindi Sub-dataset (BHSig-H)**: 160 signers, 24 genuine and 30 skilled forgeries per signer.
- **Language/Script**: Indic (Bengali, Devanagari).
- **Primary Use**: Cross-script generalization and non-Latin character stroke verification.

### 2.3 ChiSig
- **Signatures**: Over 10,000 Chinese signature samples with genuine and forge sets.
- **Language/Script**: Hanzi / Simplified and Traditional Chinese.
- **Primary Use**: High-density stroke signature verification.

## 3. Data Split & Preprocessing Standards
- **Train / Validation / Test Ratio**: 70% / 15% / 15% writer-disjoint splitting (signers in test set never appear in training set).
- **Input Dimensions**:
  - Full documents for detection: Resized to 1024x1024 with aspect ratio preservation and padding.
  - Signature crops for verification: Resized to 224x224, normalized to ImageNet mean/std `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`.
