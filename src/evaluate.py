"""Regression metrics and comparison to the naive baseline."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def regression_metrics(y_true, y_pred):
    """MAE (main), RMSE. Ignores pairs where either side is NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "n": 0}

    err = y_true[mask] - y_pred[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(mask.sum())}


def mae_improvement(model_mae, naive_mae):
    """Fraction MAE dropped vs naive. Positive = better than baseline."""
    if naive_mae == 0 or not np.isfinite(naive_mae) or not np.isfinite(model_mae):
        return np.nan
    return float((naive_mae - model_mae) / naive_mae)


def classification_metrics(y_true, y_label, y_score=None):
    """F1 and confusion matrix for the binary label; ROC-AUC if scores are given.

    Naive baseline has only hard labels, so pass y_score=None for it.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_label = np.asarray(y_label, dtype=int)
    out = {
        "f1": float(f1_score(y_true, y_label, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_label, labels=[0, 1]).tolist(),
        "positive_rate": float(y_true.mean()),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = np.nan
    return out
