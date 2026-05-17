"""
Preprocessing Pipelines Module for Seizure Prediction.

This module provides two distinct preprocessing pipelines designed to analyze
epileptic signals. It contains custom scikit-learn transformers to support:
1. Bandpass filtering (noise/artifact removal in raw time-series domains).
2. Time-series feature extraction (mean, std, max, min, energy).
3. Dimensionality reduction (PCA) and feature selection (SelectKBest).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, lfilter
from typing import Tuple, Union

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates coefficients for a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz (e.g., 0.5Hz to remove DC offset/sweat artifacts).
    highcut : float
        Upper cutoff frequency in Hz (e.g., 40Hz to exclude high-frequency muscle noise).
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, by default 5.

    Returns
    -------
    b, a : Tuple[np.ndarray, np.ndarray]
        Filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(
    data: np.ndarray, 
    lowcut: float = 0.5, 
    highcut: float = 40.0, 
    fs: float = 256.0, 
    order: int = 5
) -> np.ndarray:
    """
    Applies Butterworth bandpass filter to 2D or 3D EEG data arrays.

    Parameters
    ----------
    data : np.ndarray
        Array containing signals. Can be 2D (samples, times) or 3D (samples, channels, times).
    lowcut : float, optional
        Lower cutoff frequency, by default 0.5.
    highcut : float, optional
        Upper cutoff frequency, by default 40.0.
    fs : float, optional
        Sampling rate, by default 256.0.
    order : int, optional
        Filter order, by default 5.

    Returns
    -------
    filtered : np.ndarray
        Filtered EEG signals of same dimensions as data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if data.ndim == 2:
        return lfilter(b, a, data, axis=-1)
    elif data.ndim == 3:
        # Loop through channels per sample to filter the third dimension
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                filtered[i, j, :] = lfilter(b, a, data[i, j, :])
        return filtered
    return data

class BandpassFilterTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn wrapper transformer for applying bandpass filtering to EEG time series.
    """
    def __init__(self, lowcut: float = 0.5, highcut: float = 40.0, fs: float = 256.0, order: int = 5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'BandpassFilterTransformer':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return apply_bandpass_filter(X, self.lowcut, self.highcut, self.fs, self.order)

class FeatureExtractorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to extract statistical and physical features from time series signals.
    Extracts: Mean, Standard Deviation, Max, Min, and Signal Energy.
    """
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'FeatureExtractorTransformer':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            # Simple single-channel signals (samples, times)
            features = np.column_stack([
                np.mean(X, axis=1),
                np.std(X, axis=1),
                np.max(X, axis=1),
                np.min(X, axis=1),
                np.sum(X**2, axis=1) # Energy of signal
            ])
            return features
        elif X.ndim == 3:
            # Multi-channel signals (samples, channels, times)
            samples, channels, times = X.shape
            all_features = []
            for i in range(channels):
                ch_data = X[:, i, :]
                ch_feat = np.column_stack([
                    np.mean(ch_data, axis=1),
                    np.std(ch_data, axis=1),
                    np.max(ch_data, axis=1),
                    np.min(ch_data, axis=1),
                    np.sum(ch_data**2, axis=1)
                ])
                all_features.append(ch_feat)
            # Concatenate features from all channels (samples, channels * 5)
            return np.concatenate(all_features, axis=1)
        return X

def get_pipeline_a(n_features: int = 10) -> Pipeline:
    """
    Pipeline A Design: Normalization -> Noise removal -> Feature selection.
    
    *Critique:* Scaling noise/outliers first distorts the actual signal values.
    """
    return Pipeline([
        ('normalize', MinMaxScaler()),
        ('noise_removal', BandpassFilterTransformer()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=n_features))
    ])

def get_pipeline_b(n_components: Union[int, float] = 0.95) -> Pipeline:
    """
    Pipeline B Design: Feature extraction -> Scaling -> PCA.
    
    *Critique:* Highly recommended because features are extracted in raw domain first, 
    then scaled together correctly, and projected to principal components.
    """
    return Pipeline([
        ('feature_extraction', FeatureExtractorTransformer()),
        ('scaling', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])
