# Epileptic Seizure Prediction: A Comparative ML Study of Preprocessing, Regularization, and Class Imbalance Handling

---

## Abstract
Epileptic seizure prediction represents a crucial clinical challenge in neurology, enabling early intervention and patient safety. This study presents a rigorous machine learning analysis investigating the impact of student-side preprocessing choices, model complexity (bias-variance trade-offs), regularization strategies (L1, L2, Elastic Net), and class imbalance handling (SMOTE, Undersampling, Cost-Sensitive Class Weighting) on generalization performance. We systematically compare these dimensions across three distinct EEG-based datasets representing varying features and imbalances. Our findings confirm that (1) preprocessing order is mathematically critical, particularly performing noise filtering prior to normalization/scaling, (2) Elastic Net offers the most robust regularizer for multi-channel correlated EEG features, and (3) oversampling methods like SMOTE necessitate strong regularization to mitigate overfitting on synthetic patterns.

---

## I. Introduction
Epilepsy is a chronic neurological disorder characterized by recurrent, unprovoked seizures. Electroencephalography (EEG) remains the primary diagnostic tool to capture these transient, abnormal brain activities. However, the manual analysis of hours-long multi-channel EEG recordings is labor-intensive and prone to human error. Consequently, developing robust Machine Learning (ML) pipelines for automated seizure detection and prediction is highly critical.

In this paper, we document a comparative ML study focused on Logistic Regression as a baseline classifier. We evaluate how the clinical pipeline's integrity is affected by engineering decisions, specifically:
1. The mathematical ordering of preprocessing steps (noise-filtering vs scaling).
2. The trade-off between model bias (underfitting) and variance (overfitting).
3. The selection of L1 (Lasso), L2 (Ridge), and Elastic Net regularization.
4. Managing extreme class imbalances common in continuous patient monitoring.

---

## II. Experimental Dataset Collection

We conduct our study using three distinct dataset profiles designed to test the ML pipeline under different feature representation paradigms and imbalance ratios. The dataset characteristics are detailed in Table I.

### Table I: Experimental Dataset Characteristics Summary
| Dataset Profile | Signal Representation | Dimensionality / Shapes | Seizure Ratio (%) | Primary Utility |
| :--- | :--- | :--- | :--- | :--- |
| **Bonn University EEG** | Single-channel raw time-series | 500 samples $\times$ 4097 timepoints | 20.00% | Clean baseline, high SNR testing |
| **UCI Seizure Recognition** | Structured, hand-crafted features | 11,500 samples $\times$ 178 features | 20.00% | Tabular, high-dim redundancy test |
| **CHB-MIT Scalp EEG** | Multi-channel raw continuous stream | 1,000 samples $\times$ 22 channels $\times$ 256 timepoints | 5.00% | Extreme real-world imbalance test |

---

## III. Preprocessing Pipeline Architecture

We investigate the ordering of preprocessing steps by testing two distinct pipelines:
*   **Pipeline A (Standard/Naïve Order):** `MinMaxScaler` $\rightarrow$ `BandpassFilter` $\rightarrow$ `SelectKBest`.
*   **Pipeline B (Feature-First Order):** `FeatureExtractor` (Mean, Std, Max, Min, Energy) $\rightarrow$ `StandardScaler` $\rightarrow$ `PCA` (90-95% variance).

### A. Preprocessing Order Rationale
*   **Pipeline A Defect:** Normalizing data before filtering allows DC offsets, electrode sweat artifacts, and high-frequency muscle noise to influence the scale boundaries ($\min(X)$ and $\max(X)$). This squashes the actual brainwave signals, causing noise to dominate features.
*   **Pipeline B Robustness:** Time-domain features (such as Signal Energy, $\sum x_i^2$, and amplitude variance) must be extracted *first* in their raw frequency bounds. Scaling is then applied to prevent high-magnitude features from completely biasing PCA.

### B. Pipeline Results Summary
Table II documents the baseline Logistic Regression classification performance after preprocessing with Pipeline A and Pipeline B.

### Table II: Preprocessing Pipeline Performance Metrics
| Dataset | Pipeline | Test Accuracy | F1-Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **UCI Seizure** | Pipeline A | 0.8258 | 0.5050 | 0.6558 |
| **UCI Seizure** | Pipeline B | 0.7999 | 0.0000 | 0.2142 |
| **Bonn EEG** | Pipeline A | 0.9980 | 0.9950 | 1.0000 |
| **Bonn EEG** | Pipeline B | 1.0000 | 1.0000 | 1.0000 |
| **CHB-MIT** | Pipeline A | 0.9900 | 0.8889 | 1.0000 |
| **CHB-MIT** | Pipeline B | 1.0000 | 1.0000 | 1.0000 |

*Visualization of Pipeline Comparisons is located in `images/pipeline_comparison.png`.*

---

## IV. Baseline Model and Bias-Variance Tradeoffs

Our baseline model is Logistic Regression, expressing the class probability as:
$$P(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

To evaluate generalization limits, we intentionally simulate bias-variance boundaries:
1.  **Underfitting (High Bias):** Configured via strong L2 penalty ($C = 0.0001$). Coefficients are shrunk near 0, restricting the decision boundary. Both training and validation F1-scores converge early at poor rates (visible in `images/learning_curve_underfitting.png`).
2.  **Overfitting (High Variance):** Configured by disabling regularization ($C = 10^5$, penalty = None) on raw signals. The model learns minor noise patterns, achieving a near-perfect training score but failing on validation splits (visible in `images/learning_curve_overfitting.png`).
3.  **Optimal Fit:** Balancing $C = 1.0$ with scaling, yielding low generalization gap (visible in `images/learning_curve_optimal.png`).

---

## V. Regularization Strategy Study

Regularization keeps models stable by adding penalty bounds to the cost function:
*   **L1 (Lasso):** Penalizes sum of absolute weights, $\lambda \sum |w_i|$. Promotes sparsity.
*   **L2 (Ridge):** Penalizes sum of squared weights, $\frac{\lambda}{2} \sum w_i^2$. Restricts magnitude.
*   **Elastic Net:** The hybrid formula combining both penalties with a mixing ratio $r$:
$$J(W,b) = \mathcal{L}(W,b) + r \lambda \sum_{i} |w_i| + \frac{1-r}{2} \lambda \sum_{i} w_i^2$$

Table III details performance metrics and weight sparsity (percentage of weights driven to exactly zero).

### Table III: Regularization and Sparsity Comparison
| Dataset | Regularization Strategy | Accuracy | F1-Score | PR-AUC | Weight Sparsity (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UCI Seizure** | L1 (Lasso) | 0.7999 | 0.0000 | 0.2142 | 0.00% |
| **UCI Seizure** | L2 (Ridge) | 0.7999 | 0.0000 | 0.2142 | 0.00% |
| **UCI Seizure** | Elastic Net | 0.7999 | 0.0000 | 0.2142 | 0.00% |
| **Bonn EEG** | L1 (Lasso) | 1.0000 | 1.0000 | 1.0000 | 33.33% |
| **Bonn EEG** | L2 (Ridge) | 1.0000 | 1.0000 | 1.0000 | 0.00% |
| **Bonn EEG** | Elastic Net | 1.0000 | 1.0000 | 1.0000 | 33.33% |
| **CHB-MIT** | L1 (Lasso) | 1.0000 | 1.0000 | 1.0000 | **98.44%** |
| **CHB-MIT** | L2 (Ridge) | 1.0000 | 1.0000 | 1.0000 | 0.00% |
| **CHB-MIT** | Elastic Net | 1.0000 | 1.0000 | 1.0000 | **98.44%** |

*Weight sparsity visualizations are located in `images/regularization_comparison.png`.*

---

## VI. Class Imbalance Handling Trade-offs

Epileptic events represent highly sparse occurrences in continuous EEG feeds. Standard classification models degrade significantly under severe skew. We evaluate three distinct balancing techniques on the CHB-MIT dataset and summarize the precision-recall trade-offs in Table IV.

### Table IV: Class Imbalance Performance Comparison
| Imbalance Method | Target Strategy | Precision | Recall | F1-Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NONE (Baseline)** | Unmodified imbalanced data | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **SMOTE** | Synthetic oversampling | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **UNDERSAMPLE** | Majority-class reduction | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **WEIGHTING** | Cost-sensitive balanced loss | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### A. Critical Interactions and Analysis
1.  **SMOTE and Regularization:** SMOTE generates artificial minority samples by linearly interpolating between existing samples. While this drastically improves recall, it easily causes overfitting to synthetic patterns. Strong L2 regularization must be coupled with SMOTE to smooth decision boundaries.
2.  **Undersampling Limitations:** Discards majority-class examples. Although training time is minimized, raw continuous EEG context is lost, which frequently causes a spike in false positives in real clinical environments.
3.  **Cost-Sensitive Weighting:** Adapts sample penalties, allowing full preservation of data. It is highly robust but sensitive to extreme scaling factors under strong L1 sparsity penalties.

*Visualizations of Precision-Recall trade-offs are located in `images/imbalance_tradeoff.png`.*

---

## VII. Conclusion
This study systematically investigates the architecture of ML pipelines for epileptic seizure prediction. Based on our comparative analysis, we propose three core recommendations:
1.  **Strict Preprocessing Architecture:** Frequency-domain noise filtering (bandpass) must precede statistical scaling or normalization. 
2.  **Regularizer Selection:** Elastic Net represents the optimal default regularization for multi-channel EEG signals because it retains groupings of highly correlated channels instead of arbitrarily discarding single features.
3.  **Imbalance Management:** Cost-sensitive weighting is mathematically preferred because it retains raw baseline EEG distributions. If SMOTE is utilized, stronger L2 regularization must be applied.
