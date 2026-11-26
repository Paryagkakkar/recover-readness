import pandas as pd
import pytest

from src.features import build_feature_table


def test_targets_and_lags_follow_each_participants_calendar():
    daily = pd.DataFrame(
        [
            ("p01", "2026-01-04", 8, 400),
            ("p02", "2026-01-01", 9, 900),
            ("p01", "2026-01-01", 4, 100),
            ("p01", "2026-01-03", None, 300),
            ("p02", "2026-01-02", 6, 600),
            ("p01", "2026-01-02", 7, 200),
        ],
        columns=["user_id", "date", "readiness", "steps"],
    )

    features = build_feature_table(daily)

    assert features["user_id"].tolist() == ["p01", "p01", "p02"]
    assert features["date"].dt.day.tolist() == [1, 3, 1]
    assert features["target_readiness"].tolist() == [7, 8, 6]
    assert features["ready_high_intensity"].tolist() == [1, 1, 0]
    assert features["day_of_week"].tolist() == [3, 5, 3]
    assert features["steps_lag1"].iloc[1] == 200
    assert features["steps_lag1"].iloc[[0, 2]].isna().all()
    assert "target_readiness" not in daily


def test_missing_calendar_day_is_neither_tomorrow_nor_yesterday():
    daily = pd.DataFrame(
        {
            "user_id": ["p01"] * 3,
            "date": ["2026-01-01", "2026-01-03", "2026-01-04"],
            "readiness": [4, 7, 8],
        }
    )

    features = build_feature_table(daily)

    assert features["date"].dt.day.tolist() == [3]
    assert features["target_readiness"].tolist() == [8]
    assert features["readiness_lag1"].isna().all()


def test_duplicate_participant_dates_are_rejected():
    daily = pd.DataFrame(
        {
            "user_id": ["p01", "p01"],
            "date": ["2026-01-01", "2026-01-01"],
            "readiness": [4, 7],
        }
    )

    with pytest.raises(ValueError, match="one row per participant per day"):
        build_feature_table(daily)
