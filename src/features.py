"""Build a supervised table for next-day PMData readiness."""

import pandas as pd


READY_THRESHOLD = 7
LAG_COLUMNS = (
    "readiness",
    "fatigue",
    "mood",
    "sleep_duration_h",
    "sleep_quality",
    "soreness",
    "stress",
    "srpe_load",
    "resting_hr",
    "steps",
    "minutes_asleep",
    "sleep_score",
    "very_active_minutes",
)
ONE_DAY = pd.Timedelta(days=1)


def build_feature_table(daily):
    """Return one training example per day with a known next-day score.

    Keep the current day's inputs, add yesterday's values and the weekday,
    and derive the binary label from tomorrow's 0–10 readiness score.
    """
    table = daily.copy()
    table["date"] = pd.to_datetime(table["date"]).dt.normalize()
    if table[["user_id", "date"]].isna().any().any():
        raise ValueError("Every daily row needs a participant and date.")
    if table.duplicated(["user_id", "date"]).any():
        raise ValueError("Expected one row per participant per day.")

    table = table.sort_values(["user_id", "date"]).reset_index(drop=True)
    by_user = table.groupby("user_id", sort=False)
    previous_day = table["date"] - by_user["date"].shift(1) == ONE_DAY
    next_day = by_user["date"].shift(-1) - table["date"] == ONE_DAY

    # Keep all calendar rows until the shifts are computed so that dropping a
    # missing target cannot change the previous day used by another example.
    for column in LAG_COLUMNS:
        if column in table.columns:
            table[f"{column}_lag1"] = by_user[column].shift(1).where(previous_day)
    table["day_of_week"] = table["date"].dt.dayofweek
    table["target_readiness"] = by_user["readiness"].shift(-1).where(next_day)

    scored = table.dropna(subset=["target_readiness"]).copy()
    scored["ready_high_intensity"] = (
        scored["target_readiness"] >= READY_THRESHOLD
    ).astype(int)
    return scored.reset_index(drop=True)
