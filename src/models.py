"""Classical models and the naive baseline.

Start simple: tomorrow's readiness ≈ today's readiness. Real models
(linear, forest, boosting) come next and must beat this.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import LAG_COLUMNS, READY_THRESHOLD


def naive_predict(X, today_col="readiness"):
    """Baseline: predict tomorrow's score as today's readiness.

    `X` is a DataFrame of feature rows (must include `today_col`).
    """
    if today_col not in X.columns:
        raise KeyError(
            f"Naive baseline needs column {today_col!r} (today's readiness)"
        )
    return X[today_col].to_numpy(dtype=float)


def naive_label(X, today_col="readiness"):
    """Naive class prediction: ready tomorrow if today's readiness >= threshold."""
    return (naive_predict(X, today_col) >= READY_THRESHOLD).astype(float)


def feature_columns(df):
    """Model inputs: today's values, yesterday's values (lag1), and day of week.

    Everything here is known at the end of today, so nothing comes from the
    future. The target and the binary label are never included.
    """
    today = [c for c in LAG_COLUMNS if c in df.columns]
    lagged = [f"{c}_lag1" for c in today if f"{c}_lag1" in df.columns]
    return today + lagged + ["day_of_week"]


def _preprocessing():
    # Imputer and scaler live inside the Pipeline, so they are fitted on the
    # training rows only (leakage rule 5). Median fill; add_indicator keeps a
    # "was missing" flag because a missing report is itself informative.
    return [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]


def build_linear_pipeline():
    """Linear regression for next-day readiness (0-10)."""
    return Pipeline(_preprocessing() + [("model", LinearRegression())])


def build_logistic_pipeline():
    """Logistic regression for the ready / prefer-recovery label.

    class_weight="balanced" because only about a fifth of days are "ready".
    """
    return Pipeline(_preprocessing() + [
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
