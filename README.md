# Seizure Prediction ML Assignment

This project explores the effects of preprocessing, model complexity, and regularization on seizure prediction.

## 1. Dataset Collection

We have selected three diverse EEG-based datasets to ensure robustness:

### A. UCI Epileptic Seizure Recognition Dataset
- **Justification**: 
    - **Size**: 11,500 samples (23.5 seconds each, sampled at 174Hz).
    - **Class Imbalance**: Original dataset has 5 classes (Seizure, Tumor area, Healthy area, Eyes closed, Eyes open). Often binarized (Seizure vs Rest).
    - **Feature Characteristics**: Extracted features (178 dimensions) representing the time series.
    - **Why**: Excellent for testing model complexity on structured, preprocessed data.

### B. CHB-MIT Scalp EEG Database
- **Justification**:
    - **Size**: Large-scale (24 pediatric subjects, hundreds of hours of recording).
    - **Class Imbalance**: Highly imbalanced (seizures are rare events).
    - **Feature Characteristics**: Raw time-series (scalp EEG, 22-23 channels).
    - **Why**: Represents real-world clinical scenarios with significant noise and imbalance.

### C. Bonn University EEG Dataset
- **Justification**:
    - **Size**: 500 segments (100 per set A-E).
    - **Class Imbalance**: Balanced segments.
    - **Feature Characteristics**: Clean segments of time-series data.
    - **Why**: Allows for controlled testing of "clean" signal processing vs clinical data.

## 2. Preprocessing Pipelines

- **Pipeline A**: Normalization → Noise removal (Butterworth) → Feature selection.
- **Pipeline B**: Feature extraction (STFT/Wavelets) → Scaling → PCA.

## 3. Modeling

- **Core Model**: Logistic Regression.
- **Regularization**: L1 (Lasso), L2 (Ridge), Elastic Net.
- **Imbalance Handling**: SMOTE, Undersampling, Class Weighting.
