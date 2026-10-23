"""Load PMData into one daily table: one row per participant per calendar day.

Expected layout (after unzipping pmdata.zip into data/raw/pmdata/):

    data/raw/pmdata/p01/fitbit/...   data/raw/pmdata/p01/pmsys/...   ...

Only daily-level files are used. The minute-level heart_rate.json,
calories.json and distance.json are not needed and can stay in the zip.

This module only loads and cleans. It does NOT create the next-day target or
any lag/rolling features — that belongs in features.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# PMSys timestamps are in UTC; Fitbit dates are already local.
# The participants were based in Norway, so we convert UTC -> Oslo time
# before taking the date. Otherwise a report at 00:30 local time would
# land on the previous day.
LOCAL_TZ = "Europe/Oslo"


def _utc_to_local_date(timestamps):
    """Convert UTC timestamp strings to local calendar dates (as datetime64 at midnight)."""
    local = pd.to_datetime(timestamps, utc=True).dt.tz_convert(LOCAL_TZ)
    return local.dt.tz_localize(None).dt.normalize()


# --- PMSys (self-reported) -------------------------------------------------

def load_wellness(user_dir):
    """Morning wellness report. `readiness` is 0-10; the others are 1-5 (3 = normal)."""
    df = pd.read_csv(user_dir / "pmsys" / "wellness.csv")
    df["date"] = _utc_to_local_date(df["effective_time_frame"])

    # A few days have two reports; keep the last one submitted that day.
    df = df.sort_values("effective_time_frame").drop_duplicates("date", keep="last")

    columns = ["date", "readiness", "fatigue", "mood", "sleep_duration_h",
               "sleep_quality", "soreness", "stress"]
    return df[columns]


def load_training_load(user_dir):
    """Daily training load from session RPE: sum of (perceived exertion x minutes)."""
    path = user_dir / "pmsys" / "srpe.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "srpe_load", "n_sessions"])

    df = pd.read_csv(path)
    df["date"] = _utc_to_local_date(df["end_date_time"])
    df["load"] = df["perceived_exertion"] * df["duration_min"]
    daily = df.groupby("date").agg(srpe_load=("load", "sum"),
                                   n_sessions=("load", "size"))
    return daily.reset_index()


# --- Fitbit -----------------------------------------------------------------

def _read_json(path):
    with open(path) as f:
        return json.load(f)


def load_daily_fitbit_value(user_dir, name):
    """Files like very_active_minutes.json: one {"dateTime", "value"} entry per day."""
    path = user_dir / "fitbit" / f"{name}.json"
    if not path.exists():
        return pd.DataFrame(columns=["date", name])

    df = pd.DataFrame(_read_json(path))
    df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
    df[name] = pd.to_numeric(df["value"])
    return df[["date", name]]


def load_resting_heart_rate(user_dir):
    """Fitbit's daily resting heart rate. A value of 0 means Fitbit could not compute it."""
    path = user_dir / "fitbit" / "resting_heart_rate.json"
    if not path.exists():
        return pd.DataFrame(columns=["date", "resting_hr"])

    records = _read_json(path)
    df = pd.DataFrame({
        "date": [pd.to_datetime(r["dateTime"]).normalize() for r in records],
        "resting_hr": [r["value"]["value"] for r in records],
    })
    df["resting_hr"] = df["resting_hr"].replace(0, np.nan)
    return df


def load_steps(user_dir):
    """steps.json is per minute; sum it to a daily total."""
    df = pd.DataFrame(_read_json(user_dir / "fitbit" / "steps.json"))
    df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
    df["steps"] = pd.to_numeric(df["value"])
    return df.groupby("date", as_index=False)["steps"].sum()


def load_sleep(user_dir):
    """Main sleep of each night, dated by the morning you wake up (Fitbit's dateOfSleep)."""
    records = [r for r in _read_json(user_dir / "fitbit" / "sleep.json") if r.get("mainSleep")]
    df = pd.DataFrame({
        "date": [pd.to_datetime(r["dateOfSleep"]) for r in records],
        "minutes_asleep": [r["minutesAsleep"] for r in records],
        "minutes_awake": [r["minutesAwake"] for r in records],
        "sleep_efficiency": [r["efficiency"] for r in records],
    })
    return df.drop_duplicates("date", keep="last")


def load_sleep_score(user_dir):
    """Fitbit sleep score (0-100). The timestamp is the wake-up time."""
    df = pd.read_csv(user_dir / "fitbit" / "sleep_score.csv")
    # Already local time despite the "Z" suffix, so we just take the date part.
    df["date"] = pd.to_datetime(df["timestamp"].str[:10])
    df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
    df = df.rename(columns={"overall_score": "sleep_score",
                            "resting_heart_rate": "sleep_resting_hr"})
    return df[["date", "sleep_score", "deep_sleep_in_minutes",
               "sleep_resting_hr", "restlessness"]]


# --- Combine ----------------------------------------------------------------

def load_user(user_dir):
    """Merge every source for one participant onto a continuous daily calendar."""
    user_dir = Path(user_dir)
    tables = [
        load_wellness(user_dir),
        load_training_load(user_dir),
        load_resting_heart_rate(user_dir),
        load_steps(user_dir),
        load_sleep(user_dir),
        load_sleep_score(user_dir),
    ]
    for name in ["sedentary_minutes", "lightly_active_minutes",
                 "moderately_active_minutes", "very_active_minutes"]:
        tables.append(load_daily_fitbit_value(user_dir, name))

    df = tables[0]
    for table in tables[1:]:
        df = df.merge(table, on="date", how="outer")

    # Reindex to every calendar day between the first and last wellness report.
    # Missing days become NaN rows instead of disappearing. This matters for
    # leakage rule 1: with no gaps, shift(-1) really means "tomorrow" and not
    # "the next day this person happened to report".
    wellness_dates = tables[0]["date"]
    all_days = pd.date_range(wellness_dates.min(), wellness_dates.max(), freq="D")
    df = df.set_index("date").reindex(all_days).rename_axis("date").reset_index()

    # Days with no training session logged have zero load, not unknown load.
    df[["srpe_load", "n_sessions"]] = df[["srpe_load", "n_sessions"]].fillna(0)

    df.insert(0, "user_id", user_dir.name)
    return df


def load_daily_table(pmdata_dir):
    """All participants stacked into one table, sorted by user_id then date."""
    user_dirs = sorted(p for p in Path(pmdata_dir).glob("p[0-9][0-9]") if p.is_dir())
    if not user_dirs:
        raise FileNotFoundError(
            f"No participant folders (p01, p02, ...) in {pmdata_dir}. "
            "Download pmdata.zip and unzip it there - see README.")

    df = pd.concat([load_user(d) for d in user_dirs], ignore_index=True)
    return df.sort_values(["user_id", "date"]).reset_index(drop=True)
