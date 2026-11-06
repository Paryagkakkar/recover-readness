"""Time-aware train/test splits.

No random splits. Default: within each user, earlier dates train and later
dates test. Held-out-users split can be added later for comparison.
"""

import pandas as pd


def time_based_split(df, user_col="user_id", date_col="date", test_fraction=0.2):
    """Per-user chronological split: earlier rows → train, later rows → test.

    `test_fraction` is the share of each user's rows kept for test (at least 1
    when the user has 2+ rows). Users with a single row go entirely to train
    so we never invent a test point with no history context.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1 (exclusive)")

    train_parts = []
    test_parts = []

    for _, group in df.groupby(user_col, sort=False):
        group = group.sort_values(date_col)
        n = len(group)
        if n < 2:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_fraction)))
        n_test = min(n_test, n - 1)  # keep at least one train row
        cut = n - n_test
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])

    train = pd.concat(train_parts, ignore_index=True)
    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else df.iloc[0:0].copy()
    )
    return train, test


def assert_no_future_in_train(train, test, user_col="user_id", date_col="date"):
    """Fail loudly if any user has a train date on/after their earliest test date."""
    if test.empty:
        return
    train_max = train.groupby(user_col)[date_col].max()
    test_min = test.groupby(user_col)[date_col].min()
    shared = train_max.index.intersection(test_min.index)
    bad = [
        u for u in shared
        if train_max.loc[u] >= test_min.loc[u]
    ]
    if bad:
        raise AssertionError(
            f"Time leakage: train dates overlap or follow test for users {bad}"
        )
