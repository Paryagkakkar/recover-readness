"""Chronological train and test split for each participant."""

from math import ceil

import pandas as pd


def time_based_split(examples, test_fraction=0.2):
    """Hold out each participant's latest examples and purge boundary labels."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    train_parts = []
    test_parts = []
    for _, participant in examples.groupby("user_id"):
        ordered = participant.sort_values("date")
        test_count = ceil(len(ordered) * test_fraction)
        test = ordered.tail(test_count)
        earlier = ordered.iloc[:-test_count]

        # The label arrives a day after the input. Keep labels strictly
        # before the first day whose inputs are used to make a test prediction.
        label_dates = earlier["date"] + pd.Timedelta(days=1)
        train = earlier.loc[label_dates < test["date"].iloc[0]]
        if not train.empty:
            train_parts.append(train)
            test_parts.append(test)

    if not train_parts:
        raise ValueError("No participant has enough days for a train and test split.")
    return pd.concat(train_parts), pd.concat(test_parts)
