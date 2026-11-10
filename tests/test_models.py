"""Sanity checks for the linear / logistic pipelines (toy data, no zip needed)."""

import numpy as np
import pandas as pd

from src.evaluate import classification_metrics
from src.models import (build_linear_pipeline, build_logistic_pipeline,
                        feature_columns, naive_label)


def _toy():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "readiness": rng.integers(0, 11, n).astype(float),
        "readiness_lag1": rng.integers(0, 11, n).astype(float),
        "stress": rng.integers(1, 6, n).astype(float),
        "day_of_week": rng.integers(0, 7, n),
        "target_readiness": rng.integers(0, 11, n).astype(float),
        "ready_high_intensity": rng.integers(0, 2, n),
    })


def test_feature_columns_never_include_target_or_label():
    cols = feature_columns(_toy())
    assert "target_readiness" not in cols
    assert "ready_high_intensity" not in cols
    assert "readiness" in cols and "readiness_lag1" in cols


def test_imputer_learns_median_from_train_only():
    df = _toy()
    cols = feature_columns(df)
    train, test = df.iloc[:40].copy(), df.iloc[40:].copy()
    train.loc[train.index[:5], "stress"] = np.nan
    test["stress"] = 999.0  # extreme test values must not leak into train fit
    pipe = build_linear_pipeline().fit(train[cols], train["target_readiness"])
    fitted = pipe.named_steps["impute"].statistics_[cols.index("stress")]
    assert fitted == train["stress"].median()
    pipe.predict(test[cols])  # transform on unseen data works


def test_pipelines_handle_missing_values():
    df = _toy()
    df.loc[df.index[:10], "stress"] = np.nan
    cols = feature_columns(df)
    assert np.isfinite(build_linear_pipeline().fit(df[cols], df["target_readiness"]).predict(df[cols])).all()
    build_logistic_pipeline().fit(df[cols], df["ready_high_intensity"])


def test_naive_label_and_classification_metrics():
    X = pd.DataFrame({"readiness": [8.0, 3.0, 7.0, 6.0]})
    labels = naive_label(X)
    np.testing.assert_array_equal(labels, [1, 0, 1, 0])
    m = classification_metrics([1, 0, 0, 0], labels)
    assert m["confusion"] == [[2, 1], [0, 1]]
    assert m["positive_rate"] == 0.25
