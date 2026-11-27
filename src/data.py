"""Load the PMData dataset into a clean daily table.

One row = one participant + one calendar day.
This module only loads and cleans data.
It does NOT create targets or lag features (that is done in features.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Participants lived in Norway. Convert UTC timestamps to local time
# so late-night reports stay on the correct calendar day.
LOCAL_TZ = "Europe/Oslo"

ACTIVITY_FILES = [
    "sedentary_minutes",
    "lightly_active_minutes",
    "moderately_active_minutes",
    "very_active_minutes",
]


def _empty_daily_df(value_col: str) -> pd.DataFrame:
    """Return an empty daily table with typed date and value columns."""
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        value_col: pd.Series(dtype="float64"),
    })


def _utc_to_local_date(timestamps) -> pd.Series:
    """Convert UTC timestamps to local calendar dates at midnight."""
    local = pd.to_datetime(timestamps, utc=True).dt.tz_convert(LOCAL_TZ)
    return local.dt.tz_localize(None).dt.normalize()


# ---------------------------------------------------------------------
# PMSys (self-reported wellness + training load)
# ---------------------------------------------------------------------

def load_wellness(user_dir: Path) -> pd.DataFrame:
    """Load morning wellness report (readiness is 0-10; other scores 1-5)."""
    path = user_dir / "pmsys" / "wellness.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required wellness file: {path}")

    df = pd.read_csv(path)
    df["date"] = _utc_to_local_date(df["effective_time_frame"])
    df = df.sort_values("effective_time_frame").drop_duplicates("date", keep="last")

    cols = [
        "date", "readiness", "fatigue", "mood",
        "sleep_duration_h", "sleep_quality", "soreness", "stress",
    ]
    return df[cols].sort_values("date").reset_index(drop=True)


def load_training_load(user_dir: Path) -> pd.DataFrame:
    """Compute daily training load as sum of (RPE * duration_min)."""
    path = user_dir / "pmsys" / "srpe.csv"
    empty = pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "srpe_load": pd.Series(dtype="float64"),
        "n_sessions": pd.Series(dtype="int64"),
    })
    if not path.exists():
        return empty

    df = pd.read_csv(path)
    if df.empty:
        return empty

    df["date"] = _utc_to_local_date(df["end_date_time"])
    df["load"] = pd.to_numeric(df["perceived_exertion"]) * pd.to_numeric(df["duration_min"])

    daily = (
        df.groupby("date", as_index=False)
        .agg(srpe_load=("load", "sum"), n_sessions=("load", "size"))
    )
    daily["n_sessions"] = daily["n_sessions"].astype("int64")
    return daily


# ---------------------------------------------------------------------
# Fitbit helpers
# ---------------------------------------------------------------------

def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_daily_fitbit_value(user_dir: Path, name: str) -> pd.DataFrame:
    """Load simple daily Fitbit JSON files such as very_active_minutes.json."""
    path = user_dir / "fitbit" / f"{name}.json"
    if not path.exists():
        return _empty_daily_df(name)

    records = _read_json(path)
    if not records:
        return _empty_daily_df(name)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
    df[name] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", name]].drop_duplicates("date", keep="last")


def load_resting_heart_rate(user_dir: Path) -> pd.DataFrame:
    """Load daily resting HR; replace 0 with NaN since Fitbit uses 0 when missing."""
    path = user_dir / "fitbit" / "resting_heart_rate.json"
    if not path.exists():
        return _empty_daily_df("resting_hr")

    records = _read_json(path)
    if not records:
        return _empty_daily_df("resting_hr")

    df = pd.DataFrame({
        "date": [pd.to_datetime(r["dateTime"]).normalize() for r in records],
        "resting_hr": [r.get("value", {}).get("value", np.nan) for r in records],
    })
    df["resting_hr"] = pd.to_numeric(df["resting_hr"], errors="coerce").replace(0, np.nan)
    return df.drop_duplicates("date", keep="last")


def load_steps(user_dir: Path) -> pd.DataFrame:
    """Aggregate minute-level Fitbit steps into daily totals."""
    path = user_dir / "fitbit" / "steps.json"
    if not path.exists():
        return _empty_daily_df("steps")

    records = _read_json(path)
    if not records:
        return _empty_daily_df("steps")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
    df["steps"] = pd.to_numeric(df["value"], errors="coerce")
    return df.groupby("date", as_index=False)["steps"].sum()


def load_sleep(user_dir: Path) -> pd.DataFrame:
    """Load main sleep records dated by wake-up date (dateOfSleep)."""
    cols = ["date", "minutes_asleep", "minutes_awake", "sleep_efficiency"]
    path = user_dir / "fitbit" / "sleep.json"
    if not path.exists():
        return pd.DataFrame(columns=cols)

    records = [r for r in _read_json(path) if r.get("mainSleep")]
    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame({
        "date": pd.to_datetime([r["dateOfSleep"] for r in records]).normalize(),
        "minutes_asleep": pd.to_numeric([r.get("minutesAsleep") for r in records], errors="coerce"),
        "minutes_awake": pd.to_numeric([r.get("minutesAwake") for r in records], errors="coerce"),
        "sleep_efficiency": pd.to_numeric([r.get("efficiency") for r in records], errors="coerce"),
    })
    return df[cols].drop_duplicates("date", keep="last")


def load_sleep_score(user_dir: Path) -> pd.DataFrame:
    """Load Fitbit sleep score metrics (0-100 scale)."""
    cols = [
        "date", "sleep_score", "deep_sleep_in_minutes",
        "sleep_resting_hr", "restlessness",
    ]
    path = user_dir / "fitbit" / "sleep_score.csv"
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["date"] = pd.to_datetime(df["timestamp"].astype(str).str[:10]).dt.normalize()
    df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
    df = df.rename(columns={
        "overall_score": "sleep_score",
        "resting_heart_rate": "sleep_resting_hr",
    })
    return df[cols]


# ---------------------------------------------------------------------
# Combine everything for one user and across participants
# ---------------------------------------------------------------------

def load_user(user_dir: Path) -> pd.DataFrame:
    """Merge all sources for one participant onto a continuous daily calendar."""
    user_dir = Path(user_dir)
    wellness = load_wellness(user_dir)
    if wellness.empty:
        raise ValueError(f"Wellness data is empty for participant {user_dir.name}.")

    tables = [
        wellness,
        load_training_load(user_dir),
        load_resting_heart_rate(user_dir),
        load_steps(user_dir),
        load_sleep(user_dir),
        load_sleep_score(user_dir),
    ]
    for name in ACTIVITY_FILES:
        tables.append(load_daily_fitbit_value(user_dir, name))

    df = tables[0]
    for table in tables[1:]:
        df = df.merge(table, on="date", how="outer")

    # Fill every day from first to last wellness report so shift(-1)
    # in feature engineering lines up with actual calendar days.
    all_days = pd.date_range(wellness["date"].min(), wellness["date"].max(), freq="D")
    df = (
        df.set_index("date")
        .reindex(all_days)
        .rename_axis("date")
        .reset_index()
    )

    df["srpe_load"] = df["srpe_load"].fillna(0.0)
    df["n_sessions"] = df["n_sessions"].fillna(0).astype("int64")
    df.insert(0, "user_id", user_dir.name)
    return df


def load_daily_table(pmdata_dir: str | Path) -> pd.DataFrame:
    """Load all pXX participant folders into one sorted daily table."""
    pmdata_dir = Path(pmdata_dir)
    user_dirs = sorted(p for p in pmdata_dir.glob("p[0-9][0-9]") if p.is_dir())

    if not user_dirs:
        raise FileNotFoundError(
            f"No participant folders (p01, p02, ...) found in {pmdata_dir}. "
            "Download the PMData dataset and extract it there (see README)."
        )

    frames = [load_user(user_dir) for user_dir in user_dirs]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["user_id", "date"]).reset_index(drop=True)