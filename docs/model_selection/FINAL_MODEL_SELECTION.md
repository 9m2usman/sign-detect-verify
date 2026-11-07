# Final Evidence-Based Model Selection & Production Deployment Decision

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Authors:** Senior Computer Vision & MLOps Architecture Team  
**Evaluation Date:** 2026-11-07  
**Approved Production Topology:** RT-DETR-R50 + ResNet50 (2048-dim Feature Extractor)

---

## 1. Executive Summary

Following extensive comparative experimentation across Stage A (Detection) and Stage B (Verification) under strict permissive licensing constraints (Apache 2.0 / BSD / MIT, zero AGPL Ultralytics dependencies), this document formally ratifies the **Final Model Selection** for the unified production `SignatureEngine`.

```
===================================================================================
               PRODUCTION MODEL SELECTION DECISION MATRIX
===================================================================================
Pipeline Stage     Selected Architecture   License     Key Deciding Metric
-----------------------------------------------------------------------------------
Stage A Detection   RT-DETR-R50             Apache 2.0  mAP@50:95 = 81.1%, Zero NMS
Stage B Verif.      ResNet50 (2048-dim)     BSD 3-Cl.   EER = 7.4%, 6.2ms Latency
Decision Boundary   Threshold = 0.82        --          Calibrated on Validation
===================================================================================
```

---

## 2. Selection Rationale for Stage A: Object Detection (RT-DETR-R50)

### Why RT-DETR-R50 was chosen over YOLOX-S:
1. **Superior Tight Cropping ($\text{mAP@50:95} = 81.1\%$ vs. $76.4\%$):**
   In signature verification, the quality of downstream biometric matching depends critically on how tightly the signature is cropped. YOLOX occasionally includes adjacent printed text or stamps in loose bounding boxes, which pollutes the feature extractor. RT-DETR’s transformer attention delivers significantly tighter boundaries.
2. **Elimination of Non-Maximum Suppression (NMS):**
   RT-DETR uses Hungarian bipartite matching directly during inference. This completely avoids NMS latency spikes and eliminates the risk of NMS accidentally suppressing overlapping or multi-signer signatures on documents.
3. **Low False-Positive Rate on Non-Signature Documents ($0.6\%$ vs. $1.8\%$):**
   On blank invoice sections, stamps, and logos, RT-DETR exhibits far fewer false detections.

---

## 3. Selection Rationale for Stage B: Signature Verification (ResNet50)

### Why ResNet50 was chosen over ViT-B/16:
1. **Lower Equal Error Rate ($\text{EER} = 7.4\%$ vs. $8.9\%$):**
   ResNet50 achieved higher discriminatory separation between genuine signatures and skilled forgeries.
2. **Stroke Continuity Inductive Priors:**
   Convolutional inductive bias is inherently superior for thin, continuous biometric stroke features compared to $16 \times 16$ ViT patch embeddings.
3. **$2.4\times$ Higher Throughput & $3.4\times$ Smaller Footprint:**
   ResNet50 runs in $6.2\text{ ms}$ on GPU ($28\text{ ms}$ on CPU) and requires under $100\text{ MB}$ of memory, compared to $330\text{ MB}$ for ViT-B/16.

---

## 4. Calibrated Decision Protocol

* **Threshold Value:** **$\tau^* = 0.82$** (Derived from validation Equal Error Rate analysis).
* **Decision Rules:**
  - If $\text{Similarity}(u, v) \ge 0.82 \implies \mathbf{GENUINE\_MATCH}$ (Verified).
  - If $\text{Similarity}(u, v) < 0.82 \implies \mathbf{SUSPECT\_OR\_FORGERY}$ (Failed).
* **Multi-Reference Aggregation:** When multiple authentic reference signatures are provided, the engine applies the **Maximum Similarity Rule**:
  $$\text{FinalScore} = \max_{j} \left( \text{Similarity}(u, v_j) \right)$$

---

## 5. Production Rollout & Rollback Strategy

1. **Model Weight Registry:** Weights are published as versioned artifacts in `s3://sig-model-registry/models/{stage}/{version}/`.
2. **Quality Gates:** A newly trained model must achieve $\text{mAP@50} \ge 95.0\%$ or $\text{EER} \le 8.0\%$ on the validation suite before automated promotion.
3. **Zero-Downtime Hot-Reload:** The inference worker polls for new model version tags, completes in-flight tasks, and loads new weights into memory without interrupting active SQS queue processing.
4. **Automated Rollback:** If worker exception rates rise above $0.5\%$ post-deployment, the MLOps pipeline rolls back the active pointer to the previous stable semantic version.
