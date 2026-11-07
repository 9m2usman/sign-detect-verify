# Technical Guide: Deep Feature Embeddings & Cosine Similarity for Signature Verification

**Project:** Signature Detection, Verification & Automated MLOps Platform  
**Stage:** Stage B - Signature Verification Pipeline  
**Mathematical Foundation:** Metric Learning, Vector Spaces & Decision Threshold Calibration

---

## 1. Mathematical Formulation of Signature Verification

Signature verification is formulated as a 1:1 biometric matching problem. Given a cropped candidate signature $I_q$ extracted from a document and a known authentic reference signature $I_r$, the objective is to determine whether $I_q$ and $I_r$ originate from the same genuine signer:

$$f(I_q, I_r) \in \{\text{GENUINE\_MATCH}, \text{SUSPECT\_OR\_FORGERY}\}$$

---

## 2. Feature Extraction & Embedding Projection

Rather than comparing raw pixel grids (which are highly sensitive to illumination, pen thickness, and slight translations), deep neural networks map signature images into a dense latent metric space:

$$\phi: \mathcal{I} \to \mathbb{R}^d$$

* **ResNet50 Backbone:** Maps $224 \times 224 \times 3$ signature crops to a **$d = 2048$**-dimensional embedding vector extracted from the global average pooling layer.
* **ViT-B/16 Backbone:** Projects $16 \times 16$ non-overlapping patches and extracts the **$d = 768$**-dimensional $[CLS]$ token from the final transformer encoder block.

### L2-Normalization
To eliminate variations caused by image contrast and stroke intensity, raw embedding vectors are projected onto the unit hypersphere:

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2} = \frac{\mathbf{e}}{\sqrt{\sum_{i=1}^d e_i^2}}$$

After L2-normalization, $\|\hat{\mathbf{e}}\|_2 = 1.0$.

---

## 3. Cosine Similarity Metric

The similarity between the query embedding $\hat{\mathbf{e}}_q$ and the reference embedding $\hat{\mathbf{e}}_r$ is computed as the inner product between the two unit vectors:

$$\text{CosSim}(\hat{\mathbf{e}}_q, \hat{\mathbf{e}}_r) = \hat{\mathbf{e}}_q \cdot \hat{\mathbf{e}}_r = \sum_{i=1}^d \hat{e}_{q, i} \cdot \hat{e}_{r, i}$$

Because the vectors are on the unit sphere, $\text{CosSim} \in [-1.0, 1.0]$. For intuitive interpretation in business logic and APIs, this is rescaled to the $[0.0, 1.0]$ interval:

$$S_{\text{norm}}(\hat{\mathbf{e}}_q, \hat{\mathbf{e}}_r) = \frac{\text{CosSim}(\hat{\mathbf{e}}_q, \hat{\mathbf{e}}_r) + 1.0}{2.0} \in [0.0, 1.0]$$

---

## 4. Multi-Reference Aggregation Strategy

When multiple authentic reference signatures $\{I_{r, 1}, I_{r, 2}, \dots, I_{r, K}\}$ are registered for a signer (e.g., standard banking KYC profiles), natural human intra-signer variation occurs across different days and writing surfaces.

Our engine applies the **Maximum Similarity Rule**:

$$S^*(I_q) = \max_{k \in \{1, \dots, K\}} S_{\text{norm}}(\hat{\mathbf{e}}_q, \hat{\mathbf{e}}_{r, k})$$

This strategy accommodates natural signer variability while preventing false rejections of legitimate signers.

---

## 5. Threshold Calibration & Operating Characteristics

```
   False Rejection (FRR)               False Acceptance (FAR)
       [ Authentic Rejected ]             [ Forgery Accepted ]
               \                               /
                \                             /
                 \                           /
                  \                         /
                   \                       /
                    \                     /
                     \       EER         /
                      \       *         /
                       \     / \       /
                        \   /   \     /
                         \ /     \   /
                          X       \ /
                         / \       X
                        /   \     / \
                       /     \   /   \
                      /       \ /     \
                     /         *       \
       0.0 --------+-----------+--------+-------- 1.0
                  0.70       0.825     0.90
                         Threshold (τ)
```

1. **Equal Error Rate (EER):** The exact threshold $\tau^*$ where $\text{FAR}(\tau^*) = \text{FRR}(\tau^*)$.
2. **Validation Calibration:** Evaluated on the validation partition, the optimal threshold was determined at **$\tau^* = 0.825$** (achieving an EER of $7.4\%$ with ResNet50).
3. **Decision Function:**
   $$\text{Decision}(I_q) = \begin{cases} \text{GENUINE\_MATCH} & \text{if } S^*(I_q) \ge 0.82 \\ \text{SUSPECT\_OR\_FORGERY} & \text{if } S^*(I_q) < 0.82 \end{cases}$$
