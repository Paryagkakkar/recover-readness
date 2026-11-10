"""Naive baseline + metric helpers."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import mae_improvement, regression_metrics
from src.models import naive_predict


def test_naive_predict_uses_today_readiness():
    X = pd.DataFrame({"readiness": [3.0, 7.0, 5.0], "other": [1, 2, 3]})
    pred = naive_predict(X)
    np.testing.assert_array_equal(pred, [3.0, 7.0, 5.0])


def test_regression_metrics_mae_rmse():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx((4 / 3) ** 0.5)


def test_mae_improvement_positive_when_better():
    assert mae_improvement(model_mae=0.8, naive_mae=1.0) == pytest.approx(0.2)
