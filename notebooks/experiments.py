"""
Epileptic Seizure Prediction Experimental Suite.

This script executes the complete set of machine learning experiments for:
1. Loading and characterizing 3 diverse EEG datasets.
2. Comparing Preprocessing Pipeline A vs. Pipeline B.
3. Simulating and plotting Overfitting vs. Underfitting scenarios.
4. Conducting a Regularization Study (L1 vs. L2 vs. Elastic Net).
5. Comparing Class Imbalance Handling techniques (SMOTE, Undersampling, Weighting).

All tables are printed as structured Pandas DataFrames and all diagnostic plots 
are automatically generated and saved to the 'images/' folder.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress scikit-learn warnings for clean output display
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.linear_model')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.pipeline')

# Set styling configurations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# Ensure source directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_loader import load_uci_dataset, load_chb_mit_subset, load_bonn_dataset
from preprocessing import get_pipeline_a, get_pipeline_b
from models import get_logistic_regression_model, calculate_metrics, handle_imbalance
from analysis import plot_learning_curve, plot_regularization_comparison, plot_imbalance_comparison

def run_all_experiments():
    # Make sure output image folder exists
    os.makedirs('images', exist_ok=True)
    
    print("=" * 60)
    print("1. LOADING DATASETS & ANALYZING CHARACTERISTICS")
    print("=" * 60)
    X_uci, y_uci = load_uci_dataset(synthetic=False)
    X_bonn, y_bonn = load_bonn_dataset(synthetic=False)
    X_chb, y_chb = load_chb_mit_subset(synthetic=False)

    dataset_summary = pd.DataFrame({
        'Dataset': ['UCI Seizure Recognition', 'Bonn University EEG', 'CHB-MIT Scalp EEG'],\
        'Total Samples': [X_uci.shape[0], X_bonn.shape[0], X_chb.shape[0]],
        'Raw Dimensions': [X_uci.shape[1:], X_bonn.shape[1:], X_chb.shape[1:]],
        'Seizure Samples': [np.sum(y_uci), np.sum(y_bonn), np.sum(y_chb)],
        'Class Imbalance Ratio (%)': [
            (np.sum(y_uci) / len(y_uci)) * 100,
            (np.sum(y_bonn) / len(y_bonn)) * 100,
            (np.sum(y_chb) / len(y_chb)) * 100
        ],
        'Feature Type': ['Tabular / Extracted', 'Single-Channel Time-Series', 'Multi-Channel (22) Time-Series']
    })
    print(dataset_summary.to_string(index=False))
    print("\n")

    print("=" * 60)
    print("2. EVALUATING PREPROCESSING PIPELINES (A vs. B)")
    print("=" * 60)
    
    def evaluate_pipeline(name, X, y, pipe_a, pipe_b):
        X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
        
        # Pipeline A
        X_a = pipe_a.fit_transform(X_flat, y)
        model_a = get_logistic_regression_model(penalty='l2')
        model_a.fit(X_a, y)
        metrics_a = calculate_metrics(y, model_a.predict(X_a), model_a.predict_proba(X_a)[:, 1])
        
        # Pipeline B
        X_b = pipe_b.fit_transform(X, y)
        model_b = get_logistic_regression_model(penalty='l2')
        model_b.fit(X_b, y)
        metrics_b = calculate_metrics(y, model_b.predict(X_b), model_b.predict_proba(X_b)[:, 1])
        
        return {
            'Dataset': name,
            'Pipe A Acc': metrics_a['accuracy'], 'Pipe A F1': metrics_a['f1_score'], 'Pipe A PR-AUC': metrics_a['pr_auc'],
            'Pipe B Acc': metrics_b['accuracy'], 'Pipe B F1': metrics_b['f1_score'], 'Pipe B PR-AUC': metrics_b['pr_auc']
        }

    res_uci = evaluate_pipeline("UCI Seizure", X_uci, y_uci, get_pipeline_a(n_features=20), get_pipeline_b(n_components=0.95))
    res_bonn = evaluate_pipeline("Bonn EEG", X_bonn, y_bonn, get_pipeline_a(n_features=5), get_pipeline_b(n_components=0.90))
    res_chb = evaluate_pipeline("CHB-MIT", X_chb, y_chb, get_pipeline_a(n_features=10), get_pipeline_b(n_components=0.90))
    
    pipeline_df = pd.DataFrame([res_uci, res_bonn, res_chb])
    print(pipeline_df.to_string(index=False))
    
    # Save Pipeline comparisons
    plot_data = []
    for r in [res_uci, res_bonn, res_chb]:
        plot_data.append({'Dataset': r['Dataset'], 'Pipeline': 'Pipeline A', 'PR-AUC': r['Pipe A PR-AUC'], 'F1': r['Pipe A F1']})
        plot_data.append({'Dataset': r['Dataset'], 'Pipeline': 'Pipeline B', 'PR-AUC': r['Pipe B PR-AUC'], 'F1': r['Pipe B F1']})
    df_plot = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=df_plot, x='Dataset', y='PR-AUC', hue='Pipeline', ax=axes[0], palette='muted')
    axes[0].set_title('PR-AUC Comparison')
    axes[0].set_ylim(0, 1.05)
    sns.barplot(data=df_plot, x='Dataset', y='F1', hue='Pipeline', ax=axes[1], palette='muted')
    axes[1].set_title('F1 Score Comparison')
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('images/pipeline_comparison.png', dpi=300)
    plt.close()
    print("Pipeline comparison plots saved to 'images/pipeline_comparison.png'.\n")

    print("=" * 60)
    print("3. GENERATING BIAS-VARIANCE (LEARNING) CURVES")
    print("=" * 60)
    X_uci_prep = get_pipeline_b(n_components=0.95).fit_transform(X_uci, y_uci)
    
    # Underfitting
    model_under = get_logistic_regression_model(penalty='l2', C=0.0001)
    plot_learning_curve(model_under, X_uci_prep, y_uci, title="Learning Curve: Underfitting (C=0.0001)")
    plt.savefig('images/learning_curve_underfitting.png', dpi=300)
    plt.close()

    # Overfitting
    model_over = get_logistic_regression_model(penalty='l2', C=1e5, solver='saga')
    plot_learning_curve(model_over, X_uci_prep, y_uci, title="Learning Curve: Overfitting (C=1e5)")
    plt.savefig('images/learning_curve_overfitting.png', dpi=300)
    plt.close()

    # Optimal
    model_opt = get_logistic_regression_model(penalty='l2', C=1.0)
    plot_learning_curve(model_opt, X_uci_prep, y_uci, title="Learning Curve: Optimal Model (C=1.0)")
    plt.savefig('images/learning_curve_optimal.png', dpi=300)
    plt.close()
    print("Bias-Variance plots saved to 'images/learning_curve_*.png'.\n")

    print("=" * 60)
    print("4. REGULARIZATION PERFORMANCE & SPARSITY STUDY")
    print("=" * 60)
    
    X_bonn_prep = get_pipeline_b(n_components=0.90).fit_transform(X_bonn, y_bonn)
    X_chb_prep = get_pipeline_b(n_components=0.90).fit_transform(X_chb, y_chb)

    print("--- Dataset Features Fed to Machine Learning Models ---")
    print(f"UCI Seizure: Raw Shape = {X_uci.shape} -> Preprocessed Features Shape = {X_uci_prep.shape} ({X_uci_prep.shape[1]} features)")
    print(f"Bonn EEG:    Raw Shape = {X_bonn.shape} -> Preprocessed Features Shape = {X_bonn_prep.shape} ({X_bonn_prep.shape[1]} features)")
    print(f"CHB-MIT:     Raw Shape = {X_chb.shape}  -> Preprocessed Features Shape = {X_chb_prep.shape} ({X_chb_prep.shape[1]} features)")
    print("-" * 60 + "\n")

    def run_reg_study(name, X, y):
        results = []
        for p in ['l1', 'l2', 'elasticnet']:
            model = get_logistic_regression_model(penalty=p, l1_ratio=0.5 if p=='elasticnet' else None)
            model.fit(X, y)
            coefs = model.coef_.flatten()
            zero_coef = np.sum(np.abs(coefs) < 1e-5)
            sparsity = (zero_coef / len(coefs)) * 100
            metrics = calculate_metrics(y, model.predict(X), model.predict_proba(X)[:, 1])
            results.append({
                'Dataset': name, 'Regularizer': p.upper(),
                'Accuracy': metrics['accuracy'], 'F1-Score': metrics['f1_score'],
                'PR-AUC': metrics['pr_auc'], 'Sparsity (%)': sparsity
            })
        return results

    reg_results = []
    reg_results.extend(run_reg_study("UCI Seizure", X_uci_prep, y_uci))
    reg_results.extend(run_reg_study("Bonn EEG", X_bonn_prep, y_bonn))
    reg_results.extend(run_reg_study("CHB-MIT", X_chb_prep, y_chb))
    
    reg_df = pd.DataFrame(reg_results)
    print(reg_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=reg_df, x='Dataset', y='PR-AUC', hue='Regularizer', ax=axes[0], palette='deep')
    axes[0].set_title('PR-AUC Across Regularizers')
    axes[0].set_ylim(0, 1.05)
    sns.barplot(data=reg_df, x='Dataset', y='Sparsity (%)', hue='Regularizer', ax=axes[1], palette='deep')
    axes[1].set_title('Model Weight Sparsity (%)')
    axes[1].set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('images/regularization_comparison.png', dpi=300)
    plt.close()
    print("Regularization plots saved to 'images/regularization_comparison.png'.\n")

    print("=" * 60)
    print("5. CLASS IMBALANCE HANDLING COMPARISONS (ON CHB-MIT)")
    print("=" * 60)
    from sklearn.metrics import precision_score, recall_score
    
    methods = ['none', 'smote', 'undersample', 'weighting']
    imbalance_results = []
    
    for m in methods:
        if m == 'weighting':
            model = get_logistic_regression_model(penalty='l2')
            model.class_weight = 'balanced'
            X_res, y_res = X_chb_prep, y_chb
        else:
            model = get_logistic_regression_model(penalty='l2')
            X_res, y_res = handle_imbalance(X_chb_prep, y_chb, method=m)
            
        model.fit(X_res, y_res)
        y_pred = model.predict(X_chb_prep)
        y_prob = model.predict_proba(X_chb_prep)[:, 1]
        
        prec = precision_score(y_chb, y_pred, zero_division=0)
        rec = recall_score(y_chb, y_pred, zero_division=0)
        metrics = calculate_metrics(y_chb, y_pred, y_prob)
        
        imbalance_results.append({
            'Method': m.upper(), 'Precision': prec, 'Recall': rec,
            'F1-Score': metrics['f1_score'], 'PR-AUC': metrics['pr_auc']
        })
        
    imbalance_df = pd.DataFrame(imbalance_results)
    print(imbalance_df.to_string(index=False))
    
    df_melted = imbalance_df.melt(id_vars='Method', value_vars=['Precision', 'Recall', 'F1-Score', 'PR-AUC'], var_name='Metric', value_name='Value')
    plot_imbalance_comparison(df_melted)
    plt.savefig('images/imbalance_tradeoff.png', dpi=300)
    plt.close()
    print("Class imbalance plots saved to 'images/imbalance_tradeoff.png'.\n")
    print("All experiments completed successfully.")

if __name__ == '__main__':
    run_all_experiments()
