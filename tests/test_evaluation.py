import numpy as np
import pandas as pd
import pytest

from src.evaluate import class_balance_summary, improvement_over_naive, score_predictions, worst_error_rows
from src.models import feature_columns, gradient_boosting_regressor
from src.split import held_out_users_split, time_based_split


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


def test_held_out_users_split_keeps_users_disjoint():
    rows = pd.DataFrame(
        {
            "user_id": ["p01"] * 3 + ["p02"] * 3 + ["p03"] * 3 + ["p04"] * 3 + ["p05"] * 3,
            "date": list(pd.date_range("2026-01-01", periods=3)) * 5,
        }
    )
    train, test = held_out_users_split(rows, test_user_fraction=0.2)
    assert set(train["user_id"]).isdisjoint(set(test["user_id"]))
    assert set(test["user_id"]) == {"p05"}


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


def test_improvement_balance_and_worst_errors():
    naive = {"mae": 1.2, "rmse": 1.5}
    model = {"mae": 0.9, "rmse": 1.2}
    gain = improvement_over_naive(model, naive)
    assert gain["mae_gain"] == pytest.approx(0.3)

    bal = class_balance_summary([1, 0, 1, 1])
    assert bal["positives"] == 3
    assert bal["positive_rate"] == pytest.approx(0.75)

    test_df = pd.DataFrame({
        "user_id": ["p01", "p02"],
        "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "readiness": [6.0, 7.0],
        "target_readiness": [4.0, 7.5],
    })
    worst = worst_error_rows(test_df, [8.0, 7.5], top_n=1)
    assert len(worst) == 1
    assert worst["user_id"].iloc[0] == "p01"
    assert worst["abs_error"].iloc[0] == pytest.approx(4.0)


def test_gradient_boosting_regressor_fits_the_demo_inputs():
    readings = pd.DataFrame(
        {
            "readiness": [1, 2, 3, 4, 6, 7, 8, 9],
            "sleep_duration_h": [5, 5, 6, 6, 7, 8, 8, 9],
            "sleep_quality": [1, 2, 2, 3, 3, 4, 4, 5],
            "resting_hr": [70, 68, 66, 64, 60, 58, 56, 54],
            "steps": [3000, 4000, 5000, 6000, 8000, 9000, 10000, 11000],
        }
    )
    name, model = gradient_boosting_regressor()
    model.fit(readings, readings["readiness"] + 0.2)
    predicted = model.predict(readings.iloc[[0]])

    assert name in {"XGBoost", "Gradient boosting"}
    assert predicted.shape == (1,)