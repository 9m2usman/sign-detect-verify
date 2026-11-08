# Data Storage and Protection Guide

## Overview

This repository handles signature detection and verification datasets comprising scanned documents, isolated signatures, bounding box coordinates, and writer identity mappings.

## Strict Protection Protocol

1. **Uncompressed and Compressed Master Datasets**:
   - `merged_signature_dataset/` (3.47 GB, 57,918 files)
   - `merged_signature_dataset.zip` (3.26 GB archive)
   These master datasets are preserved strictly in place at the root directory of the repository and must never be altered, overwritten, deleted, or committed to Git.

2. **Git Tracking Policy**:
   - All directories under `data/raw/` and `data/processed/` are untracked by Git via `.gitignore` except their respective `.gitkeep` markers.
   - Raw document scans, full-resolution signature crops, generated pair combinations, and embeddings must strictly remain in local storage or in S3 buckets.

## Directory Structure

```
data/
├── README.md               # This documentation and protection policy
├── raw/                    # Raw incoming or extracted images (untracked)
└── processed/              # Processed crops, preprocessed tensors, metadata manifests (untracked)
```

## Dataset Composition Summary

- **Total Images / Files**: 57,918 files across verified public datasets.
- **Included Benchmarks**:
  - CEDAR (genuine and forged signature pairs)
  - BHSig260 (Bengali and Hindi scripts)
  - ChiSig (Chinese signatures)
  - SigComp series benchmarks
  - Synthetic document overlays
- **Permissive Licensing**:
  All models and pipelines trained from this data use strictly permissive open-source licenses (MIT/Apache 2.0/BSD). Refer to `LICENSES.md` and `docs/research_references.md` for complete attributions and citations.
