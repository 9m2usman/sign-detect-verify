# Phase 2 Deliverable: Verification Dataset Comparative Research & Recommendation (Dataset B)

**Research Scope:** Signature Verification & Metric Learning  
**Status:** Research Completed & Recommendation Formulated (Awaiting User Download Approval)  

---

## 1. Candidate Datasets Comparison Matrix

| Evaluation Criteria | Candidate 1: CEDAR (Recommended) | Candidate 2: BHSig260 | Candidate 3: GPDS-960 | Candidate 4: SigComp (ICDAR) |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Writers** | **55 writers** | 260 writers (100 Bengali, 160 Hindi) | 960 writers | 10 to 30 writers per competition |
| **Signatures per Writer**| 24 genuine, 24 forgeries | 24 genuine, 30 forgeries | 24 genuine, 30 forgeries | Varies by year (10–24) |
| **Total Signature Images**| **2,640 images** | 14,040 images | 51,840 images | ~1,000 to 2,000 images |
| **Forgery Type** | **Skilled forgeries** (forgers practiced genuine signatures) | Skilled forgeries | Skilled & simple forgeries | Forensic skilled forgeries |
| **Script / Language** | Latin / English | Indic (Bengali & Devanagari) | Western (Spanish) | Multi-script |
| **Image Resolution** | 300 DPI clean scans (grayscale PNG) | Scanned at 300 DPI (TIFF/PNG) | Varying scan resolutions (JPEG) | 300 to 600 DPI |
| **Writer Identifiers** | Yes (`writer_id` from 1 to 55) | Yes (`writer_id` 1 to 260) | Yes (`writer_id` 1 to 960) | Yes |
| **Verification Labels** | `G` (Genuine) vs. `F` (Forgery) | Genuine vs. Forged | Genuine vs. Forged | Genuine vs. Forged |
| **Licensing & Access** | Academic research use (Public/Open) | Academic research use | Restrictive academic agreement | Competition benchmark |
| **Benchmark Status** | **Industry-standard gold benchmark** for offline signature verification | High diversity benchmark for non-Latin scripts | Large-scale evaluation | Annual forensic benchmark |

---

## 2. Deep Dive: CEDAR Signature Dataset

### Architectural Suitability for Siamese Metric Learning
1. **Balanced Positive & Negative Pair Generation:**
   For each of the 55 writers:
   - Genuine signatures: 24
   - Skilled forgeries: 24
   - $\text{Genuine-Genuine Pairs (Positive, label = 1):} \;\; \binom{24}{2} = 276 \text{ pairs per writer}$
   - $\text{Genuine-Forgery Pairs (Negative, label = 0):} \;\; 24 \times 24 = 576 \text{ pairs per writer}$
   - Total possible comparison pairs across 55 writers: $> 46,000$ pairs.
2. **Skilled vs. Random Forgeries:**
   Unlike random forgeries (which any basic perceptual hash or template matcher can easily distinguish), CEDAR's forgeries are created by human forgers who examined the authentic signature and attempted to mimic slant, loops, and stroke geometry. This forces the Siamese network to learn deep micro-texture and stroke topology rather than simple macroscopic shape outlines.
3. **Zero Data Leakage (Writer-Independent Split):**
   - **Training Set (Writers 1 to 40):** 40 writers (1,920 signatures) used exclusively for metric learning optimization.
   - **Validation Set (Writers 41 to 45):** 5 writers (240 signatures) used for margin tuning, contrastive loss calibration, and threshold $\tau$ selection.
   - **Test Set (Writers 46 to 55):** 10 writers (480 signatures) strictly held out. The network has never seen these writers' handwriting or signatures during training.

---

## 3. Secondary Candidate: BHSig260
BHSig260 is an exceptional secondary dataset. It contains signatures from 260 individuals written in Bengali (100 writers) and Hindi (160 writers).
- **Pros:** High diversity, tests cross-script generalization to complex curved Indic scripts.
- **Cons:** Significantly larger archive size (14,040 images), TIFF formats requiring normalization, and uneven noise baselines across scans.

---

## 4. Final Recommendation & Gate Condition

**Recommendation:** Adopt the **CEDAR Signature Dataset** as the primary Dataset B verification benchmark for Phase 3 training and evaluation.

> [!IMPORTANT]
> **Phase 2 Gate Enforcement:** In accordance with Section 3 and Section 19 of the master requirements, **the CEDAR dataset will NOT be downloaded until the user gives explicit written authorization to begin Phase 3 data preparation.**
