"""Time-split sanity checks (toy calendar, no PMData zip)."""

import pandas as pd
import pytest

from src.split import assert_no_future_in_train, time_based_split


def _toy():
    rows = []
    for user in ("p01", "p02"):
        for i in range(10):
            rows.append({
                "user_id": user,
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "readiness": i,
                "target_readiness": i + 1,
            })
    return pd.DataFrame(rows)


def test_time_split_is_chronological_per_user():
    train, test = time_based_split(_toy(), test_fraction=0.2)
    assert_no_future_in_train(train, test)
    for user in ("p01", "p02"):
        tr = train[train["user_id"] == user]
        te = test[test["user_id"] == user]
        assert tr["date"].max() < te["date"].min()
        assert len(te) == 2  # 20% of 10


def test_time_split_keeps_all_rows():
    df = _toy()
    train, test = time_based_split(df, test_fraction=0.2)
    assert len(train) + len(test) == len(df)


def test_single_row_user_goes_to_train():
    df = pd.DataFrame({
        "user_id": ["p99"],
        "date": [pd.Timestamp("2020-01-01")],
        "readiness": [5],
    })
    train, test = time_based_split(df, test_fraction=0.2)
    assert len(train) == 1
    assert len(test) == 0
