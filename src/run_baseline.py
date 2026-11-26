"""Run the readiness benchmark without saving data or trained models."""

import argparse

import pandas as pd

from src.data import load_daily_table
from src.evaluate import score_predictions
from src.features import READY_THRESHOLD, build_feature_table
from src.models import feature_columns, model_pairs
from src.split import time_based_split


def run_benchmark(data_dir):
    examples = build_feature_table(load_daily_table(data_dir))
    train, test = time_based_split(examples)
    shared_test = test.dropna(subset=["readiness"])
    if shared_test.empty:
        raise ValueError("No test days have today's readiness for the simple guess.")

    columns = feature_columns(train)
    actual = shared_test["target_readiness"]
    naive = shared_test["readiness"]
    predictions = {"Tomorrow = today": naive}
    results = {
        "Tomorrow = today": score_predictions(actual, naive, naive >= READY_THRESHOLD, naive),
    }
    for name, (regressor, classifier) in model_pairs().items():
        regressor.fit(train[columns], train["target_readiness"])
        classifier.fit(train[columns], train["ready_high_intensity"])
        predicted_scores = regressor.predict(shared_test[columns]).clip(0, 10)
        predicted_labels = classifier.predict(shared_test[columns])
        probabilities = classifier.predict_proba(shared_test[columns])[:, 1]
        predictions[name] = predicted_scores
        results[name] = score_predictions(actual, predicted_scores, predicted_labels, probabilities)

    print(f"{len(train):,} training days; {len(test):,} test days; "
          f"{len(shared_test):,} test days used for all comparisons.")
    print(f"Ready class: {train.ready_high_intensity.mean():.1%} of training days; "
          f"{shared_test.ready_high_intensity.mean():.1%} of compared test days.")
    print(pd.DataFrame.from_dict(results, orient="index").to_string(float_format="%.3f"))
    print("\nMean absolute error by participant")
    errors = pd.DataFrame(predictions, index=shared_test.index).sub(actual, axis=0).abs()
    print(errors.groupby(shared_test["user_id"]).mean().to_string(float_format="%.3f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/pmdata")
    run_benchmark(parser.parse_args().data_dir)
