"""Regression metrics and comparison to the naive baseline."""

import numpy as np


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
