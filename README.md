# Seizure Prediction ML Assignment

This project explores the effects of preprocessing, model complexity, and regularization on seizure prediction.

## 1. Dataset Collection

We have selected three diverse EEG-based datasets to ensure robustness:

### A. Epileptic Seizure Recognition Dataset
- **Function**: `load_epileptic_seizure_recognition()`
- **Justification**: 
    - **Size**: 11,500 samples.
    - **Class Imbalance**: Original dataset has 5 classes. Binarized to class 1 (seizure, 20.00%) vs classes 2-5 (non-seizure, 80.00%).
    - **Feature Characteristics**: Extracted tabular features (178 dimensions).
    - **Why**: Excellent for testing model complexity on structured, preprocessed data.

### B. BEED (Bangalore EEG Epilepsy Dataset)
- **Function**: `load_beed_dataset()`
- **Justification**:
    - **Size**: 8,000 samples.
    - **Class Imbalance**: Balanced (50.00% seizure ratio: 4,000 positive, 4,000 negative).
    - **Feature Characteristics**: Multi-channel EEG (16 channels, X1-X16).
    - **Why**: Captures clinical multi-electrode records across adult subjects, perfect for evaluating multi-channel spatial mappings.

### C. CHB-MIT Seizure Dataset (NPZ)
- **Function**: `load_chb_mit_npz()`
- **Justification**:
    - **Size**: 1,000 samples selected from the clinical validation set (`eeg-seizure_val.npz`).
    - **Class Imbalance**: Extremely imbalanced (exactly 5.00% seizure ratio: 50 positive, 950 negative) to replicate clinical rare-event scenarios.
    - **Feature Characteristics**: Raw continuous EEG multi-channel waves (23 channels $\times$ 256 timepoints).
    - **Why**: Represents realistic, high-noise clinical settings with severe imbalance, acting as our class imbalance testbed.

## 2. Preprocessing Pipelines

- **Pipeline A**: Normalization → Noise removal (Butterworth) → Feature selection.
- **Pipeline B**: Feature extraction (Mean, Std, Max, Min, Energy) → Scaling → PCA.

## 3. Modeling

- **Core Model**: Logistic Regression.
- **Regularization**: L1 (Lasso), L2 (Ridge), Elastic Net.
- **Imbalance Handling**: SMOTE, Undersampling, Class Weighting.

