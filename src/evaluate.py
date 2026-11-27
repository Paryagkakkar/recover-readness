"""Evaluation metrics and diagnostics for next-day recovery readiness."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from src.features import READY_THRESHOLD


def class_balance_summary(labels) -> dict:
    """Return count and positive rate for binary readiness labels."""
    arr = np.asarray(labels)
    total = int(arr.size)
    positives = int(np.sum(arr == 1))
    rate = float(positives / total) if total > 0 else 0.0
    return {
        "total": total,
        "positives": positives,
        "negatives": total - positives,
        "positive_rate": rate,
    }


def score_predictions(actual_scores, predicted_scores, predicted_labels, ranking_scores):
    """Compute MAE, RMSE, F1, ROC-AUC, and confusion matrix."""
    actual_arr = np.asarray(actual_scores, dtype="float64")
    pred_scores_arr = np.asarray(predicted_scores, dtype="float64")
    pred_labels_arr = np.asarray(predicted_labels, dtype="int64")
    actual_labels = (actual_arr >= READY_THRESHOLD).astype("int64")

    auc = None
    if np.unique(actual_labels).size == 2:
        auc = float(roc_auc_score(actual_labels, ranking_scores))

    return {
        "mae": float(mean_absolute_error(actual_arr, pred_scores_arr)),
        "rmse": float(root_mean_squared_error(actual_arr, pred_scores_arr)),
        "f1": float(f1_score(actual_labels, pred_labels_arr, zero_division=0)),
        "roc_auc": auc,
        "confusion": confusion_matrix(actual_labels, pred_labels_arr, labels=[0, 1]).tolist(),
    }


def improvement_over_naive(model_metrics: dict, naive_metrics: dict) -> dict:
    """Positive values indicate lower error than the 'tomorrow = today' baseline."""
    return {
        "mae_gain": float(naive_metrics["mae"] - model_metrics["mae"]),
        "rmse_gain": float(naive_metrics["rmse"] - model_metrics["rmse"]),
    }


def worst_error_rows(test_df: pd.DataFrame, predicted_scores, top_n: int = 10) -> pd.DataFrame:
    """Return the rows with largest absolute regression error for error analysis."""
    actual = pd.to_numeric(test_df["target_readiness"], errors="coerce")
    preds = pd.Series(predicted_scores, index=test_df.index, dtype="float64")
    abs_err = (preds - actual).abs()

    cols = [c for c in ["user_id", "date", "readiness", "target_readiness"] if c in test_df.columns]
    out = test_df[cols].copy()
    out["pred_readiness"] = preds
    out["abs_error"] = abs_err
    return out.sort_values("abs_error", ascending=False).head(top_n).reset_index(drop=True)