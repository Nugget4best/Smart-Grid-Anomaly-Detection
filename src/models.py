"""
Supervised and unsupervised anomaly detection models.

Implements a comprehensive model suite:
    - Isolation Forest (unsupervised)
    - Local Outlier Factor (unsupervised)
    - One-Class SVM (semi-supervised)
    - Random Forest (supervised)
    - XGBoost (supervised)
    - Ensemble voting (supervised)
"""

import time
import numpy as np
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


# ============================================================
# Unsupervised Anomaly Detection
# ============================================================

def create_isolation_forest(contamination=0.08, n_estimators=200, random_state=42):
    """
    Create Isolation Forest anomaly detector.

    Isolation Forest isolates anomalies by randomly selecting features and
    split values. Anomalies require fewer splits to isolate, yielding shorter
    path lengths in the tree ensemble.

    Reference:
        Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008).
        "Isolation Forest." ICDM.
    """
    return IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples='auto',
        random_state=random_state,
        n_jobs=-1,
    )


def create_lof(contamination=0.08, n_neighbors=20):
    """
    Create Local Outlier Factor detector.

    LOF computes the local density deviation of each sample relative to
    its neighbors. Samples with substantially lower density are flagged
    as outliers.

    Reference:
        Breunig, M.M. et al. (2000). "LOF: Identifying Density-Based
        Local Outliers." SIGMOD.
    """
    return LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=n_neighbors,
        novelty=True,
        n_jobs=-1,
    )


def create_ocsvm(kernel='rbf', gamma='scale', nu=0.08):
    """
    Create One-Class SVM for anomaly detection.

    One-Class SVM learns a boundary around the normal data distribution
    in kernel space. Points outside this boundary are classified as anomalies.

    Reference:
        Scholkopf, B. et al. (2001). "Estimating the Support of a
        High-Dimensional Distribution." Neural Computation.
    """
    return OneClassSVM(
        kernel=kernel,
        gamma=gamma,
        nu=nu,
    )


# ============================================================
# Supervised Classification
# ============================================================

def create_random_forest(n_estimators=200, max_depth=15, random_state=42, **kwargs):
    """Create Random Forest classifier with class weight balancing."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def create_xgboost(n_estimators=200, max_depth=8, learning_rate=0.1,
                    scale_pos_weight=None, random_state=42, **kwargs):
    """
    Create XGBoost classifier.

    scale_pos_weight handles class imbalance (ratio of negative to positive).
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
        **kwargs,
    )


def create_gradient_boosting(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=42):
    """Create Gradient Boosting classifier."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def create_voting_ensemble(rf_params=None, xgb_params=None, voting='soft',
                            random_state=42):
    """Create soft-voting ensemble of RF, XGBoost, and Gradient Boosting."""
    rf_params = rf_params or {}
    xgb_params = xgb_params or {}

    estimators = [
        ('rf', create_random_forest(random_state=random_state, **rf_params)),
        ('xgb', create_xgboost(random_state=random_state, **xgb_params)),
        ('gb', create_gradient_boosting(random_state=random_state)),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1,
    )


# ============================================================
# Training Utilities
# ============================================================

def train_model(model, X_train, y_train=None):
    """Train a model and return training time."""
    start = time.time()
    if y_train is not None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.2f}s")
    return model, elapsed


def train_unsupervised_models(X_train_normal, contamination=0.08):
    """
    Train unsupervised anomaly detectors on normal data only.

    For semi-supervised evaluation, these models are trained exclusively
    on normal (non-anomalous) samples and then evaluated on the full
    test set containing both normal and anomalous samples.
    """
    results = {}

    models = {
        'Isolation Forest': create_isolation_forest(contamination=contamination),
        'Local Outlier Factor': create_lof(contamination=contamination),
        'One-Class SVM': create_ocsvm(nu=contamination),
    }

    for name, model in models.items():
        print(f"\nTraining: {name}")
        trained, elapsed = train_model(model, X_train_normal)
        results[name] = {'model': trained, 'train_time': elapsed}

    return results


def train_supervised_models(X_train, y_train, scale_pos_weight=None):
    """Train all supervised classifiers."""
    results = {}

    models = {
        'Random Forest': create_random_forest(),
        'XGBoost': create_xgboost(scale_pos_weight=scale_pos_weight),
        'Gradient Boosting': create_gradient_boosting(),
        'Voting Ensemble': create_voting_ensemble(),
    }

    for name, model in models.items():
        print(f"\nTraining: {name}")
        trained, elapsed = train_model(model, X_train, y_train)
        results[name] = {'model': trained, 'train_time': elapsed}

    return results
