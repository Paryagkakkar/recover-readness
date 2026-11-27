"""Build supervised features for next-day readiness prediction (PMData)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Binary label threshold on PMData's 0-10 readiness scale (derived from target).
READY_THRESHOLD = 7
DEFAULT_TARGET_SLEEP_H = 8.0

# Columns used for lag-1 (yesterday) features when the prior calendar day exists.
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

ROLLING_BASE_COLUMNS = [
    "readiness",
    "resting_hr",
    "sleep_duration_h",
    "sleep_score",
    "srpe_load",
]

FEATURE_GROUPS = {
    "base": ["day_of_week", *LAG_COLUMNS, *(f"{c}_lag1" for c in LAG_COLUMNS)],
    "rolling": [
        *(f"{c}_roll3_mean" for c in ROLLING_BASE_COLUMNS),
        *(f"{c}_roll7_mean" for c in ROLLING_BASE_COLUMNS),
    ],
    "sleep_debt": ["sleep_hours_est", "sleep_debt_3d", "sleep_debt_7d"],
    "baseline_dev": ["resting_hr_dev_7d", "readiness_dev_7d"],
    "strain_ratio": ["acwr_srpe"],
}


def same_day_linear_r2(daily: pd.DataFrame, target_col: str = "readiness") -> float | None:
    """Check if target is nearly deterministic from same-day numeric features."""
    if target_col not in daily.columns:
        return None

    drop_cols = {"user_id", "date", target_col}
    candidate_cols = [
        c for c in daily.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(daily[c])
    ]
    if not candidate_cols:
        return None

    subset = daily[candidate_cols + [target_col]].dropna()
    if len(subset) < 10:
        return None

    model = LinearRegression()
    model.fit(subset[candidate_cols], subset[target_col])
    preds = model.predict(subset[candidate_cols])
    return float(r2_score(subset[target_col], preds))


def _sleep_hours_series(df: pd.DataFrame) -> pd.Series:
    """Prefer sleep_duration_h; fall back to minutes_asleep / 60 when unavailable."""
    if "sleep_duration_h" in df.columns:
        hours = pd.to_numeric(df["sleep_duration_h"], errors="coerce")
    else:
        hours = pd.Series(np.nan, index=df.index, dtype="float64")

    if "minutes_asleep" in df.columns:
        from_minutes = pd.to_numeric(df["minutes_asleep"], errors="coerce") / 60.0
        hours = hours.fillna(from_minutes)
    return hours


def _add_user_window_features(user_df: pd.DataFrame, target_sleep_h: float) -> pd.DataFrame:
    """Compute calendar-windowed features inside a single user's timeline."""
    out = user_df.sort_values("date").copy()
    indexed = out.set_index("date")

    for col in ROLLING_BASE_COLUMNS:
        if col in indexed.columns:
            series = pd.to_numeric(indexed[col], errors="coerce")
            out[f"{col}_roll3_mean"] = series.rolling("3D", min_periods=1).mean().to_numpy()
            out[f"{col}_roll7_mean"] = series.rolling("7D", min_periods=1).mean().to_numpy()

    sleep_h = _sleep_hours_series(out)
    out["sleep_hours_est"] = sleep_h.to_numpy()
    sleep_idx = pd.Series(sleep_h.to_numpy(), index=indexed.index)
    deficit = (target_sleep_h - sleep_idx).clip(lower=0.0)
    out["sleep_debt_3d"] = deficit.rolling("3D", min_periods=1).sum().to_numpy()
    out["sleep_debt_7d"] = deficit.rolling("7D", min_periods=1).sum().to_numpy()

    if "resting_hr" in indexed.columns:
        rhr = pd.to_numeric(indexed["resting_hr"], errors="coerce")
        rhr_7d = rhr.rolling("7D", min_periods=2).mean()
        out["resting_hr_dev_7d"] = (rhr - rhr_7d).to_numpy()

    if "readiness" in indexed.columns:
        rd = pd.to_numeric(indexed["readiness"], errors="coerce")
        rd_7d = rd.rolling("7D", min_periods=2).mean()
        out["readiness_dev_7d"] = (rd - rd_7d).to_numpy()

    if "srpe_load" in indexed.columns:
        load = pd.to_numeric(indexed["srpe_load"], errors="coerce").fillna(0.0)
        acute = load.rolling("3D", min_periods=1).mean()
        chronic = load.rolling("14D", min_periods=1).mean()
        out["acwr_srpe"] = np.where(chronic > 0.0, acute / chronic, 0.0)

    return out


def build_feature_table(
    daily: pd.DataFrame,
    target_sleep_h: float = DEFAULT_TARGET_SLEEP_H,
) -> pd.DataFrame:
    """Create one supervised row per day with a valid next-day readiness target."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if df[["user_id", "date"]].isna().any().any():
        raise ValueError("Every row must have a user_id and a date.")

    if df.duplicated(subset=["user_id", "date"]).any():
        raise ValueError(
            "Found duplicate user_id + date rows. Expected one row per participant per day."
        )

    if "readiness" not in df.columns:
        raise ValueError("Missing required 'readiness' column in daily table.")

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    user_frames = [
        _add_user_window_features(part, target_sleep_h=target_sleep_h)
        for _, part in df.groupby("user_id", sort=False)
    ]
    df = pd.concat(user_frames, ignore_index=True)
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("user_id", sort=False)
    is_prev_day_consecutive = (df["date"] - grouped["date"].shift(1)) == pd.Timedelta(days=1)
    is_next_day_consecutive = (grouped["date"].shift(-1) - df["date"]) == pd.Timedelta(days=1)

    for col in LAG_COLUMNS:
        if col in df.columns:
            df[f"{col}_lag1"] = grouped[col].shift(1).where(is_prev_day_consecutive)

    df["day_of_week"] = df["date"].dt.dayofweek.astype("int64")
    df["target_readiness"] = grouped["readiness"].shift(-1).where(is_next_day_consecutive)

    df = df.dropna(subset=["target_readiness"]).copy()
    df["ready_high_intensity"] = (df["target_readiness"] >= READY_THRESHOLD).astype("int64")

    return df.reset_index(drop=True)