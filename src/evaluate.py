"""Metrics for the score and derived readiness label."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
)

from src.features import READY_THRESHOLD


def score_predictions(actual_scores, predicted_scores, predicted_labels, ranking_scores):
    actual_labels = np.asarray(actual_scores) >= READY_THRESHOLD
    auc = None
    if np.unique(actual_labels).size == 2:
        auc = roc_auc_score(actual_labels, ranking_scores)
    return {
        "mae": mean_absolute_error(actual_scores, predicted_scores),
        "rmse": root_mean_squared_error(actual_scores, predicted_scores),
        "f1": f1_score(actual_labels, predicted_labels, zero_division=0),
        "roc_auc": auc,
        "confusion": confusion_matrix(actual_labels, predicted_labels, labels=[0, 1]).tolist(),
    }
