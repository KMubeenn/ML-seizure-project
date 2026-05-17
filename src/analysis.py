"""
Analysis and Visualization Module for the Seizure Prediction Project.

This module provides helper functions to visualize machine learning metrics,
learning curves, regularization comparisons, and class imbalance trade-offs.
It is designed to generate publication-quality figures for analysis reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Dict, Any, Union

def plot_learning_curve(
    estimator: BaseEstimator, 
    X: np.ndarray, 
    y: np.ndarray, 
    title: str = "Learning Curve"
) -> plt.Figure:
    """
    Generates a learning curve plot comparing training and cross-validation scores.
    
    This is highly useful for diagnosing:
    - Underfitting (bias): High error/low scores on both training and validation sets.
    - Overfitting (variance): Large gap between training and validation scores.

    Parameters
    ----------
    estimator : BaseEstimator
        The scikit-learn estimator instance (e.g., LogisticRegression).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    title : str, optional
        Title of the generated plot, by default "Learning Curve".

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate learning curve metrics
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Style elements
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Training Examples (Sample Size)", fontsize=12)
    ax.set_ylabel("Performance Score (Accuracy/F1)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot confidence intervals (standard deviation bounds)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # Plot means
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", linewidth=2, label="Training Score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", linewidth=2, label="Cross-Validation Score")

    ax.legend(loc="best", fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    return fig

def plot_regularization_comparison(results_dict: Dict[str, float]) -> plt.Figure:
    """
    Generates a bar plot comparing performance metrics for different regularization strategies.

    Parameters
    ----------
    results_dict : Dict[str, float]
        Dictionary mapping regularization strategies (e.g., 'L1', 'L2', 'ElasticNet')
        to their resulting performance value (F1-Score or PR-AUC).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    names = list(results_dict.keys())
    values = list(results_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=names, y=values, ax=ax, palette="Blues_d")
    
    ax.set_title("Regularization Strategy Comparison", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("F1 Score / PR-AUC", fontsize=12)
    ax.set_xlabel("Regularization Type", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    # Add value labels on top of the bars
    for idx, val in enumerate(values):
        ax.text(idx, val + 0.02, f"{val:.4f}", ha='center', fontweight='bold', fontsize=11)
        
    return fig

def plot_imbalance_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """
    Generates a grouped bar chart comparing performance metrics across 
    different class imbalance handling methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame in melted format containing columns ['Method', 'Metric', 'Value'].

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Grouped barplot
    sns.barplot(data=results_df, x='Method', y='Value', hue='Metric', ax=ax, palette='viridis')
    
    ax.set_title("Impact of Class Imbalance Handling on Model Performance", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Imbalance Handling Technique", fontsize=12)
    ax.set_ylabel("Score Value", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, facecolor='white')
    
    return fig
