"""Next-day target and lag features for PMData.

Rules this module must keep:
1. Target is tomorrow's readiness, shifted per user (never global).
2. Lag features use only today and earlier, also per user.
3. No rolling / sleep-debt / strain-ratio here yet — those come later,
   one group at a time so we can ablate them.

Does not split, scale, or train. That lives in split.py / models.py.
"""

import pandas as pd

# Wellness + Fitbit columns we lag. Keep this list short and explicit so
# a viva answer is "yesterday's value of each of these".
LAG_COLUMNS = [
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
]

# Binary label is derived from the continuous target (course honesty rule).
# >= 7 ≈ "ready for high intensity" on the 0–10 PMSys scale.
READY_THRESHOLD = 7


def add_next_day_target(df, readiness_col="readiness", user_col="user_id"):
    """Add tomorrow's readiness as the regression target; drop rows with no tomorrow.

    Also adds `ready_high_intensity` = (target >= READY_THRESHOLD). This label
    is derived from the score — say so in the report; it is not independent.
    """
    out = df.copy()
    out["target_readiness"] = out.groupby(user_col)[readiness_col].shift(-1)
    out = out.dropna(subset=["target_readiness"]).copy()
    out["target_readiness"] = out["target_readiness"].astype(float)
    out["ready_high_intensity"] = (out["target_readiness"] >= READY_THRESHOLD).astype(int)
    return out.reset_index(drop=True)


def add_lag_features(df, columns=None, user_col="user_id", lag=1):
    """Add yesterday's value of each column, computed inside groupby(user).

    Rows where yesterday is missing stay NaN (first day per user, or a gap
    after a missing report day that was filled as NaN by the loader).
    """
    if columns is None:
        columns = [c for c in LAG_COLUMNS if c in df.columns]

    out = df.copy()
    for col in columns:
        out[f"{col}_lag{lag}"] = out.groupby(user_col)[col].shift(lag)
    return out


def add_calendar_features(df, date_col="date"):
    """Day-of-week (0=Mon … 6=Sun). Safe: known at prediction time for 'today'."""
    out = df.copy()
    out["day_of_week"] = pd.to_datetime(out[date_col]).dt.day_of_week
    return out


def build_feature_table(df):
    """Loader output → modelling table: lags, calendar, then next-day target.

    Order matters: build features on the full calendar first, then shift the
    target and drop the last day per user. That way lag features on day T
    still see day T-1 even when we drop day T+1's missing target.
    """
    out = add_lag_features(df)
    out = add_calendar_features(out)
    out = add_next_day_target(out)
    return out
