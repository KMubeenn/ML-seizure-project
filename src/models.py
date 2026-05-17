"""
Model Building and Evaluation Module for Seizure Prediction.

This module provides model builders and configurations to study:
1. Standard Baseline Logistic Regression models.
2. Sparsity and Regularization (L1, L2, and Elastic Net penalties).
3. Class Imbalance methods (SMOTE, random undersampling, and weight penalty).
4. Specific overfitting and underfitting model scenarios.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional, Union

# Suppress scikit-learn SAGA solver/penalty warning outputs during execution
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.linear_model')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')

def get_logistic_regression_model(
    penalty: Optional[str] = 'l2', 
    C: float = 1.0, 
    solver: str = 'lbfgs', 
    l1_ratio: Optional[float] = None
) -> LogisticRegression:
    """
    Initializes a Logistic Regression model with custom regularization parameters.

    Parameters
    ----------
    penalty : str or None, optional
        Type of regularization: 'l1' (Lasso), 'l2' (Ridge), 'elasticnet', or None, by default 'l2'.
    C : float, optional
        Inverse of regularization strength. Small values specify stronger regularization, by default 1.0.
    solver : str, optional
        Optimization algorithm. 'saga' is required for L1 and Elastic Net, by default 'lbfgs'.
    l1_ratio : float or None, optional
        The mixing parameter for Elastic Net (r in the objective function), by default None.

    Returns
    -------
    LogisticRegression
        Configured scikit-learn Logistic Regression model.
    """
    # Map deprecated 'none' to modern Python None
    if penalty == 'none':
        penalty = None

    if penalty == 'elasticnet':
        # Elastic Net requires the SAGA solver
        return LogisticRegression(
            penalty=penalty, 
            C=C, 
            solver='saga', 
            l1_ratio=l1_ratio, 
            max_iter=1000, 
            random_state=42
        )
    elif penalty == 'l1':
        # L1 Lasso requires SAGA or liblinear solver
        return LogisticRegression(
            penalty=penalty, 
            C=C, 
            solver='saga', 
            max_iter=1000, 
            random_state=42
        )
    else:
        # Standard L2 or Unregularized
        return LogisticRegression(
            penalty=penalty, 
            C=C, 
            solver=solver, 
            max_iter=1000, 
            random_state=42
        )

def handle_imbalance(
    X: np.ndarray, 
    y: np.ndarray, 
    method: str = 'smote'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies oversampling, undersampling, or weighting to address class imbalance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Imbalanced labels.
    method : str, optional
        Imbalance method: 'smote', 'undersample', or 'none', by default 'smote'.

    Returns
    -------
    X_resampled, y_resampled : Tuple[np.ndarray, np.ndarray]
        Resampled feature matrix and labels.
    """
    if method == 'smote':
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        return rus.fit_resample(X, y)
    return X, y

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes diagnostic classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted class labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class (seizures).

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing Accuracy, F1-Score, and PR-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # PR-AUC calculation is highly robust for imbalanced sets
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'pr_auc': pr_auc
    }

def create_underfitting_scenario() -> LogisticRegression:
    """
    Builds a model representing underfitting (High Bias) via strong regularization.
    """
    return get_logistic_regression_model(penalty='l2', C=0.0001)

def create_overfitting_scenario() -> LogisticRegression:
    """
    Builds a model representing overfitting (High Variance) via disabled regularization.
    """
    return get_logistic_regression_model(penalty=None, solver='saga')
