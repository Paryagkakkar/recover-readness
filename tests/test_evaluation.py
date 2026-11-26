import numpy as np
import pandas as pd
import pytest

from src.evaluate import score_predictions
from src.models import feature_columns
from src.split import time_based_split


def test_split_keeps_training_labels_before_each_test_period():
    rows = pd.DataFrame(
        {
            "user_id": ["a"] * 10 + ["b"] * 5,
            "date": list(pd.date_range("2026-01-01", periods=10))
            + list(pd.date_range("2026-02-01", periods=5)),
        }
    )
    train, test = time_based_split(rows.sample(frac=1, random_state=2))

    assert train.groupby("user_id").size().to_dict() == {"a": 7, "b": 3}
    assert test.groupby("user_id").size().to_dict() == {"a": 2, "b": 1}
    last_training_label = train.groupby("user_id")["date"].max() + pd.Timedelta(days=1)
    assert (last_training_label < test.groupby("user_id")["date"].min()).all()


def test_model_inputs_exclude_targets_and_unreviewed_columns():
    table = pd.DataFrame(
        columns=["user_id", "date", "day_of_week", "readiness", "readiness_lag1",
                 "target_readiness", "ready_high_intensity", "future_steps"]
    )

    assert set(feature_columns(table)) == {"day_of_week", "readiness", "readiness_lag1"}


def test_score_predictions_reports_errors_and_confusion_counts():
    scores = score_predictions([8, 5, 9, 6], [8, 4, 8, 6], [1, 0, 0, 0],
                               [0.9, 0.1, 0.4, 0.2])

    assert scores["mae"] == 0.5
    assert scores["rmse"] == pytest.approx(np.sqrt(0.5))
    assert scores["f1"] == pytest.approx(2 / 3)
    assert scores["roc_auc"] == 1.0
    assert scores["confusion"] == [[2, 0], [1, 1]]
