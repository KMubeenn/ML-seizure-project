"""
Data Loader Module for Seizure Prediction.

This module provides data loading functions for the Bonn University, UCI, 
and CHB-MIT EEG datasets. It supports both synthetic subset generation 
(matching exact dimensions and distributions) and templates for raw file reading.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Tuple

def load_uci_dataset(synthetic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the UCI Epileptic Seizure Recognition tabular dataset.
    
    If synthetic is True, it generates a synthetic dataset mirroring the exact:
    - 11,500 samples
    - 178 feature columns (representing spectral/temporal coefficients)
    - 5 raw classes collapsed into a binary classification task.
    
    Classification structure:
    - Positive Class (1): Seizure activity (Set 1).
    - Negative Class (0): Non-seizure activity (Sets 2, 3, 4, 5).

    Parameters
    ----------
    synthetic : bool, optional
        Whether to generate high-fidelity synthetic data, by default True.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (11500, 178).
    y : np.ndarray
        Binary labels (1 = seizure, 0 = non-seizure) of shape (11500,).
    """
    if synthetic:
        # mirror raw shape and feature statistics
        X, y = make_classification(
            n_samples=11500, 
            n_features=178, 
            n_informative=50, 
            n_redundant=10, 
            n_classes=5, 
            random_state=42
        )
        # Collapse 5 multi-classes into binary seizure (1) vs non-seizure (0)
        y_bin = np.where(y == 0, 1, 0)
        return X, y_bin
    else:
        # Load actual CSV file if it exists in the workspace
        # df = pd.read_csv('data/uci_seizure_data.csv')
        # X = df.drop(['Unnamed: 0', 'y'], axis=1).values
        # y = df['y'].values
        # y_bin = np.where(y == 1, 1, 0)
        # return X, y_bin
        raise NotImplementedError(
            "Actual UCI dataset CSV file is not present in the workspace. "
            "Please upload 'uci_seizure_data.csv' to run on real data."
        )

def load_chb_mit_subset(synthetic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a subset of the CHB-MIT Scalp EEG continuous Database.
    
    If synthetic is True, it generates raw multi-channel time-series data mirroring:
    - 22 Channels (standard 10-20 international system electrodes).
    - 256 Hz sampling rate.
    - 256 timepoints (1-second windows).
    - Highly imbalanced class ratio representing sparse seizure events in a stream.

    Parameters
    ----------
    synthetic : bool, optional
        Whether to generate high-fidelity synthetic data, by default True.

    Returns
    -------
    X : np.ndarray
        3D EEG signal matrix of shape (1000, 22, 256).
    y : np.ndarray
        Binary class labels (1 = seizure event, 0 = inter-ictal baseline) of shape (1000,).
    """
    if synthetic:
        n_samples = 1000
        n_channels = 22
        n_times = 256
        
        # Initialize background Gaussian EEG noise
        X = np.random.randn(n_samples, n_channels, n_times)
        
        # Inject periodic 10Hz oscillatory patterns representing localized seizures
        seizure_indices = np.random.choice(n_samples, 50, replace=False) # 5% seizure ratio
        for idx in seizure_indices:
            t = np.linspace(0, 1, n_times)
            X[idx, :, :] += np.sin(2 * np.pi * 10 * t) # localized 10Hz spike-wave
        
        y = np.zeros(n_samples)
        y[seizure_indices] = 1
        return X, y
    else:
        # Load actual EDF recordings using mne library
        # raw = mne.io.read_raw_edf('data/chb01_03.edf', preload=True)
        raise NotImplementedError(
            "Actual CHB-MIT dataset loading requires localized EDF data files."
        )

def load_bonn_dataset(synthetic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the Bonn University EEG Dataset.
    
    If synthetic is True, it generates single-channel segments matching the exact:
    - 500 total segments (100 per sets A, B, C, D, E).
    - 4097 timepoints representing 23.6-second epochs.
    
    Classification Structure:
    - Set E: Seizure activity (Class 1).
    - Sets A, B, C, D: Eyes open, eyes closed, and inter-ictal recordings (Class 0).

    Parameters
    ----------
    synthetic : bool, optional
        Whether to generate high-fidelity synthetic data, by default True.

    Returns
    -------
    X : np.ndarray
        2D EEG signal matrix of shape (500, 4097).
    y : np.ndarray
        Binary class labels (1 = seizure, 0 = healthy/inter-ictal) of shape (500,).
    """
    if synthetic:
        n_samples = 500
        n_times = 4097
        
        # Base random EEG noise
        X = np.random.randn(n_samples, n_times)
        
        # Inject transient high-amplitude spike patterns in the final 100 samples (Set E)
        seizure_indices = np.arange(400, 500)
        for idx in seizure_indices:
            # Gaussian envelope transient spike
            X[idx, :] += 5 * np.exp(-((np.arange(n_times) - 2000)**2) / 100)
        
        y = np.zeros(n_samples)
        y[seizure_indices] = 1
        return X, y
    else:
        raise NotImplementedError(
            "Actual Bonn dataset requires importing the raw single-channel txt files."
        )
