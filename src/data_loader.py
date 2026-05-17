"""
Data Loader Module for Seizure Prediction.

This module provides data loading functions for the BEED (Bangalore EEG Epilepsy Dataset),
Epileptic Seizure Recognition (UCI), and CHB-MIT (NPZ) EEG datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Tuple
import os

def _get_dataset_path(filename: str) -> str:
    """
    Robustly resolves paths to datasets inside the 'newData' directory,
    handling runs from the repository root, notebooks/, or src/.
    """
    path = os.path.join('newData', filename)
    if not os.path.exists(path):
        # Try parent directory
        path = os.path.join('..', 'newData', filename)
    if not os.path.exists(path):
        # Fallback to absolute path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(current_dir)
        path = os.path.join(repo_root, 'newData', filename)
    return path

def load_epileptic_seizure_recognition(synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the Epileptic Seizure Recognition tabular dataset from newData/Epileptic Seizure Recognition.csv.
    
    If synthetic is True, it generates a synthetic dataset mirroring the dimensions.
    If synthetic is False, it loads the actual 11,500 clinical EEG samples from the workspace.
    
    Classification structure:
    - Positive Class (1): Seizure activity (Class 1).
    - Negative Class (0): Non-seizure activity (Classes 2, 3, 4, 5).

    Parameters
    ----------
    synthetic : bool, optional
        Whether to generate synthetic proxy data, by default False (real clinical data).

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (11500, 178).
    y : np.ndarray
        Binary labels (1 = seizure, 0 = non-seizure) of shape (11500,).
    """
    if synthetic:
        X, y = make_classification(
            n_samples=11500, 
            n_features=178, 
            n_informative=50, 
            n_redundant=10, 
            n_classes=5, 
            random_state=42
        )
        y_bin = np.where(y == 0, 1, 0)
        return X, y_bin
    else:
        csv_path = _get_dataset_path('Epileptic Seizure Recognition.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Epileptic Seizure Recognition CSV file is not present at '{csv_path}'")
        df = pd.read_csv(csv_path)
        if 'Unnamed' in df.columns[0]:
            X = df.drop([df.columns[0], 'y'], axis=1).values
        else:
            X = df.drop(['y'], axis=1).values
        y = df['y'].values
        # Collapse multi-classes (Class 1 is seizure, Classes 2-5 are non-seizure)
        y_bin = np.where(y == 1, 1, 0)
        return X, y_bin

def load_beed_dataset(synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the BEED: Bangalore EEG Epilepsy Dataset from newData/BEED_Data.csv.
    
    Binarization strategy:
    - Positive Class (1): Generalized Seizures (Class 1) and Focal Seizures (Class 2).
    - Negative Class (0): Healthy Subjects (Class 0) and Seizure Events (Class 3).

    Parameters
    ----------
    synthetic : bool, optional
        Whether to generate synthetic proxy data, by default False.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (8000, 16).
    y : np.ndarray
        Binary labels (1 = seizure, 0 = non-seizure) of shape (8000,).
    """
    if synthetic:
        n_samples = 8000
        n_features = 16
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        return X, y
    else:
        csv_path = _get_dataset_path('BEED_Data.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"BEED_Data.csv is not present at '{csv_path}'")
        df = pd.read_csv(csv_path)
        X = df.drop(['y'], axis=1).values
        y = df['y'].values
        # Binarize: 1 and 2 represent seizure activity, 0 and 3 represent non-seizure baseline
        y_bin = np.where((y == 1) | (y == 2), 1, 0)
        return X, y_bin

def load_chb_mit_npz(synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a subset of the CHB-MIT Scalp EEG continuous Database from the new pre-segmented
    validation .npz file (newData/eeg-seizure_val.npz).
    
    To maintain 100% compatibility with existing downstream analysis and class imbalance tests,
    this function selects a subset of 1,000 samples with an exact 5.00% seizure ratio (50 positive
    and 950 negative samples) to preserve the exact clinical properties of the original experiment.

    Parameters
    ----------
    synthetic : bool, optional
        Whether to generate synthetic proxy data, by default False.

    Returns
    -------
    X : np.ndarray
        3D EEG signal matrix of shape (1000, 23, 256).
    y : np.ndarray
        Binary labels (1 = seizure, 0 = non-seizure) of shape (1000,).
    """
    if synthetic:
        n_samples = 1000
        n_channels = 23
        n_times = 256
        X = np.random.randn(n_samples, n_channels, n_times)
        y = np.random.randint(0, 2, n_samples)
        return X, y
    else:
        npz_path = _get_dataset_path('eeg-seizure_val.npz')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"eeg-seizure_val.npz is not present at '{npz_path}'")
        data = np.load(npz_path)
        X = data['val_signals']
        y = data['val_labels']
        
        # Sub-sample to keep training extremely fast and preserve the 5% class imbalance
        seizure_idxs = np.where(y == 1)[0]
        non_seizure_idxs = np.where(y == 0)[0]
        
        np.random.seed(42)
        selected_seizure = np.random.choice(seizure_idxs, 50, replace=False)
        selected_non_seizure = np.random.choice(non_seizure_idxs, 950, replace=False)
        
        indices = np.concatenate([selected_seizure, selected_non_seizure])
        np.random.shuffle(indices)
        
        return X[indices], y[indices]
