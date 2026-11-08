# Signature Verification Evaluation Methodology

## 1. Metric Definitions

### 1.1 False Acceptance Rate (FAR)
$$\text{FAR}(\tau) = \frac{\text{Forged signatures accepted as genuine}}{\text{Total forged test pairs}}$$

### 1.2 False Rejection Rate (FRR)
$$\text{FRR}(\tau) = \frac{\text{Genuine signatures rejected as forgeries}}{\text{Total genuine test pairs}}$$

### 1.3 Equal Error Rate (EER)
The point on the ROC curve where $\text{FAR}(\tau) = \text{FRR}(\tau)$. A lower EER indicates higher verification accuracy.

### 1.4 Area Under the ROC Curve (ROC-AUC)
Measures the overall discriminative ability across all potential decision thresholds $\tau \in [0, 1]$.

## 2. Test Pair Generation Protocol
- **Genuine Pairs**: Pairs formed by pairing genuine signatures from the same writer $(g_i^{(w)}, g_j^{(w)})$ where $i \ne j$.
- **Skilled Forgery Pairs**: Pairs formed by pairing a genuine signature with a skilled forgery targeting that writer $(g_i^{(w)}, f_k^{(w)})$.
- **Random / Unskilled Forgery Pairs**: Pairs formed by pairing genuine signatures from different writers $(g_i^{(w)}, g_j^{(v)})$ where $w \ne v$.
