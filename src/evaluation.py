"""
Evaluation metrics and visualization for anomaly detection.

Provides specialized metrics for imbalanced anomaly detection:
    - Precision-Recall curves (preferred over ROC for imbalanced data)
    - F1-optimal threshold selection
    - Detection latency analysis
    - Comprehensive comparison across model families
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive anomaly detection metrics.

    Includes Matthews Correlation Coefficient (MCC) and Average Precision (AP)
    which are more robust than accuracy for imbalanced datasets.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None

    return metrics


def evaluate_unsupervised(results, X_test, y_test):
    """
    Evaluate unsupervised models.

    Unsupervised models output -1 (anomaly) or 1 (normal).
    Convert to 0/1 for consistent evaluation.
    """
    records = []
    for name, data in results.items():
        model = data['model']
        y_pred_raw = model.predict(X_test)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = np.where(y_pred_raw == -1, 1, 0)

        # Get anomaly scores if available
        y_scores = None
        if hasattr(model, 'decision_function'):
            y_scores = -model.decision_function(X_test)  # negate so higher = more anomalous
        elif hasattr(model, 'score_samples'):
            y_scores = -model.score_samples(X_test)

        metrics = compute_metrics(y_test, y_pred, y_scores)
        metrics['model'] = name
        metrics['type'] = 'unsupervised'
        metrics['train_time'] = data['train_time']
        records.append(metrics)

    return pd.DataFrame(records).set_index('model')


def evaluate_supervised(results, X_test, y_test):
    """Evaluate supervised models."""
    records = []
    for name, data in results.items():
        model = data['model']
        y_pred = model.predict(X_test)

        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['model'] = name
        metrics['type'] = 'supervised'
        metrics['train_time'] = data['train_time']
        records.append(metrics)

    return pd.DataFrame(records).set_index('model')


def plot_confusion_matrices(results, X_test, y_test, model_type='supervised',
                             figsize=None, save_path=None):
    """Plot confusion matrices for all models in a single figure."""
    n_models = len(results)
    if figsize is None:
        figsize = (5 * n_models, 4)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        model = data['model']
        y_pred_raw = model.predict(X_test)

        if model_type == 'unsupervised':
            y_pred = np.where(y_pred_raw == -1, 1, 0)
        else:
            y_pred = y_pred_raw

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle(f'{model_type.title()} Models — Confusion Matrices', fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_roc_pr_curves(results, X_test, y_test, model_type='supervised',
                        save_path=None):
    """Plot ROC and Precision-Recall curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, data in results.items():
        model = data['model']

        if model_type == 'supervised' and hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_scores = -model.decision_function(X_test)
        elif hasattr(model, 'score_samples'):
            y_scores = -model.score_samples(X_test)
        else:
            continue

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_val:.3f})')

        # PR curve
        prec, rec, _ = precision_recall_curve(y_test, y_scores)
        ap = average_precision_score(y_test, y_scores)
        ax2.plot(rec, prec, label=f'{name} (AP={ap:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    baseline = y_test.mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_reconstruction_error_distribution(errors_normal, errors_anomaly,
                                            threshold=None, save_path=None):
    """Plot reconstruction error distributions for normal vs anomaly samples."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors_normal, bins=80, alpha=0.7, label='Normal', color='steelblue', density=True)
    ax.hist(errors_anomaly, bins=80, alpha=0.7, label='Anomaly', color='coral', density=True)

    if threshold is not None:
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')

    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Reconstruction Error Distribution: Normal vs Anomaly')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_model_comparison(all_eval_df, metric_cols=None, save_path=None):
    """Plot comprehensive model comparison bar chart."""
    if metric_cols is None:
        metric_cols = ['precision', 'recall', 'f1_score', 'mcc']

    available = [c for c in metric_cols if c in all_eval_df.columns]
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(all_eval_df))
    width = 0.8 / len(available)

    colors = plt.cm.Set2(np.linspace(0, 1, len(available)))
    for i, (col, color) in enumerate(zip(available, colors)):
        vals = all_eval_df[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=col.replace('_', ' ').title(),
               color=color, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(all_eval_df.index, rotation=25, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison — All Methods')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_")
        return None

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importances[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def print_classification_report(y_true, y_pred, title=""):
    """Print formatted classification report."""
    if title:
        print(f"\n{'='*60}")
        print(f"Classification Report: {title}")
        print('=' * 60)
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Anomaly'],
                                zero_division=0))
