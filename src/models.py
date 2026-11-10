"""Classical models and the naive baseline.

Start simple: tomorrow's readiness ≈ today's readiness. Real models
(linear, forest, boosting) come next and must beat this.
"""

import numpy as np


def naive_predict(X, today_col="readiness"):
    """Baseline: predict tomorrow's score as today's readiness.

    `X` is a DataFrame of feature rows (must include `today_col`).
    """
    if today_col not in X.columns:
        raise KeyError(
            f"Naive baseline needs column {today_col!r} (today's readiness)"
        )
    return X[today_col].to_numpy(dtype=float)
