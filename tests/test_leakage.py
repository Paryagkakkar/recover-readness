"""Sanity checks for next-day target and per-user lags (no real data needed)."""

import pandas as pd

from src.features import (
    READY_THRESHOLD,
    add_lag_features,
    add_next_day_target,
    build_feature_table,
)


def _toy_daily():
    """Two users, 4 consecutive days each. Readiness = user digit + day index."""
    rows = []
    for user, base in [("p01", 3), ("p02", 5)]:
        for i in range(4):
            rows.append({
                "user_id": user,
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "readiness": base + i,
                "fatigue": 3,
                "srpe_load": float(i),
            })
    return pd.DataFrame(rows)


def test_target_is_next_day_per_user():
    df = add_next_day_target(_toy_daily())
    # Last calendar day per user has no tomorrow → dropped.
    assert len(df) == 6
    p01 = df[df["user_id"] == "p01"].sort_values("date")
    # Day 0 readiness 3 → target should be day 1's readiness 4
    assert p01.iloc[0]["target_readiness"] == 4
    assert p01.iloc[1]["target_readiness"] == 5
    assert p01.iloc[2]["target_readiness"] == 6


def test_target_never_crosses_users():
    """p01's last usable target must not become p02's first readiness."""
    df = add_next_day_target(_toy_daily())
    p01_last = df[df["user_id"] == "p01"].sort_values("date").iloc[-1]
    p02_first_readiness = _toy_daily().loc[
        (_toy_daily()["user_id"] == "p02") & (_toy_daily()["date"] == "2020-01-01"),
        "readiness",
    ].iloc[0]
    # p01 day 2 readiness 5 → target day 3 readiness 6, not p02's 5
    assert p01_last["target_readiness"] == 6
    assert p01_last["target_readiness"] != p02_first_readiness


def test_binary_label_derived_from_threshold():
    df = add_next_day_target(_toy_daily())
    assert (df["ready_high_intensity"] == (df["target_readiness"] >= READY_THRESHOLD)).all()


def test_lag_is_yesterday_per_user():
    df = add_lag_features(_toy_daily(), columns=["readiness", "srpe_load"])
    p02 = df[df["user_id"] == "p02"].sort_values("date")
    assert pd.isna(p02.iloc[0]["readiness_lag1"])
    assert p02.iloc[1]["readiness_lag1"] == 5  # yesterday = day 0
    assert p02.iloc[2]["srpe_load_lag1"] == 1.0


def test_lag_does_not_cross_users():
    """First row of p02 must not pick up p01's last readiness as lag."""
    df = add_lag_features(_toy_daily(), columns=["readiness"])
    p02_first = df[df["user_id"] == "p02"].sort_values("date").iloc[0]
    assert pd.isna(p02_first["readiness_lag1"])


def test_build_feature_table_smoke():
    out = build_feature_table(_toy_daily())
    assert "target_readiness" in out.columns
    assert "readiness_lag1" in out.columns
    assert "day_of_week" in out.columns
    assert out["target_readiness"].notna().all()
