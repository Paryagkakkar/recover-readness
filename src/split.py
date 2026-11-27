"""Train/test split helpers for daily recovery readiness data."""

from math import ceil

import pandas as pd


def _prepare_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and normalize dates."""
    required = {"user_id", "date"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"Missing required columns for splitting: {sorted(missing)}")

    df = examples.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def time_based_split(examples: pd.DataFrame, test_fraction: float = 0.2):
    """Hold out each participant's latest days and purge boundary labels."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    df = _prepare_examples(examples)
    train_parts = []
    test_parts = []

    for _, participant in df.groupby("user_id", sort=True):
        ordered = participant.sort_values("date").reset_index(drop=True)
        n_rows = len(ordered)
        test_count = ceil(n_rows * test_fraction)

        if test_count <= 0 or test_count >= n_rows:
            continue

        test = ordered.tail(test_count)
        earlier = ordered.iloc[:-test_count]
        first_test_date = test["date"].iloc[0]

        # Purge training rows whose next-day target falls on or after test starts.
        label_dates = earlier["date"] + pd.Timedelta(days=1)
        train = earlier.loc[label_dates < first_test_date]

        if not train.empty:
            train_parts.append(train)
            test_parts.append(test)

    if not train_parts:
        raise ValueError("No participant has enough days for a valid train and test split.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def held_out_users_split(examples: pd.DataFrame, test_user_fraction: float = 0.2):
    """Split participants into disjoint train and test user groups."""
    if not 0 < test_user_fraction < 1:
        raise ValueError("test_user_fraction must be between 0 and 1.")

    df = _prepare_examples(examples)
    users = sorted(df["user_id"].unique())
    if len(users) < 2:
        raise ValueError("Need at least 2 users for a held-out-users split.")

    test_count = max(1, ceil(len(users) * test_user_fraction))
    if test_count >= len(users):
        test_count = len(users) - 1

    test_users = set(users[-test_count:])
    train_df = df.loc[~df["user_id"].isin(test_users)].reset_index(drop=True)
    test_df = df.loc[df["user_id"].isin(test_users)].reset_index(drop=True)
    return train_df, test_df