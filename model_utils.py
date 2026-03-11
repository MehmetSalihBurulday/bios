"""Makine ogrenmesi modelleri icin ortak yardimci fonksiyonlar."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def generate_bios_like_data(
    n_samples: int = 1500,
    n_features: int = 20,
    random_state: int = 42,
):
    """Bilgi tasiyan ve sinif dengesizligi iceren ornek veri uretir."""
    n_informative = max(4, int(n_features * 0.6))
    n_redundant = max(2, int(n_features * 0.2))
    n_repeated = max(0, int(n_features * 0.05))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features - 2),
        n_redundant=min(n_redundant, max(1, n_features - n_informative - 1)),
        n_repeated=n_repeated,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        class_sep=1.2,
        flip_y=0.02,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int32)


def split_dataset(X, y, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """Veriyi train/validation/test olarak boler."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_summary(name, X, y):
    distribution = pd.Series(y).value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"{name}: {X.shape}, sinif dagilimi: {distribution}")


def find_best_threshold(y_true, y_scores, start: float = 0.2, stop: float = 0.8, step: float = 0.02):
    """Validation skorlari uzerinden en iyi F1 esigini bulur."""
    thresholds = np.arange(start, stop + step, step)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (y_scores >= threshold).astype(int)
        current_f1 = f1_score(y_true, predictions, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_binary_classifier(y_true, y_scores, threshold: float = 0.5):
    """Siniflandirici ciktilarini temel metriklerle degerlendirir."""
    y_scores = np.asarray(y_scores).ravel()
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "average_precision": average_precision_score(y_true, y_scores),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }