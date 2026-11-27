"""Run the readiness benchmark without saving data or trained models."""

import argparse

import numpy as np
import pandas as pd

from src.data import load_daily_table
from src.evaluate import improvement_over_naive, score_predictions, worst_error_rows
from src.features import FEATURE_GROUPS, READY_THRESHOLD, build_feature_table, same_day_linear_r2
from src.models import feature_columns, model_pairs
from src.split import held_out_users_split, time_based_split


def _positive_class_probabilities(classifier, X: pd.DataFrame) -> np.ndarray:
    """Return probability of class 1 even if training data had a single class."""
    probs = classifier.predict_proba(X)
    classes = list(getattr(classifier, "classes_", []))
    if 1 in classes:
        return probs[:, classes.index(1)]
    return np.zeros(len(X), dtype="float64")


def run_benchmark(
    data_dir: str,
    test_fraction: float = 0.2,
    split_mode: str = "time",
    ablation: bool = False,
) -> None:
    """Load PMData, run leakage-safe split, evaluate baseline and candidate models."""
    daily = load_daily_table(data_dir)

    r2 = same_day_linear_r2(daily, target_col="readiness")
    if r2 is not None:
        print(f"Same-day linear R^2 on readiness: {r2:.4f}")
        if r2 >= 0.98:
            print(
                "WARNING: Same-day R^2 is near 1.0; target may be formula-derived "
                "from same-day features."
            )

    examples = build_feature_table(daily)
    if split_mode == "users":
        train, test = held_out_users_split(examples, test_user_fraction=test_fraction)
    elif split_mode == "time":
        train, test = time_based_split(examples, test_fraction=test_fraction)
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}. Use 'time' or 'users'.")

    shared_test = test.dropna(subset=["readiness"])
    if shared_test.empty:
        raise ValueError("No test days have today's readiness for the naive baseline.")

    columns = feature_columns(train)
    actual = shared_test["target_readiness"]
    naive = shared_test["readiness"]

    naive_metrics = score_predictions(
        actual, naive, (naive >= READY_THRESHOLD).astype(int), naive
    )
    predictions = {"Tomorrow = today": naive}
    results = {"Tomorrow = today": naive_metrics}

    for name, (regressor, classifier) in model_pairs().items():
        regressor.fit(train[columns], train["target_readiness"])
        classifier.fit(train[columns], train["ready_high_intensity"])

        predicted_scores = np.clip(regressor.predict(shared_test[columns]), 0.0, 10.0)
        predicted_labels = classifier.predict(shared_test[columns])
        probabilities = _positive_class_probabilities(classifier, shared_test[columns])

        predictions[name] = predicted_scores
        results[name] = score_predictions(
            actual, predicted_scores, predicted_labels, probabilities
        )

    print(
        f"Split ({split_mode}): {len(train):,} training days; {len(test):,} test days; "
        f"{len(shared_test):,} test days used for all comparisons."
    )
    print(
        f"Ready class: {train['ready_high_intensity'].mean():.1%} of training days; "
        f"{shared_test['ready_high_intensity'].mean():.1%} of compared test days."
    )
    print(pd.DataFrame.from_dict(results, orient="index").to_string(float_format="%.3f"))

    print("\nMAE / RMSE improvement over naive baseline ('Tomorrow = today')")
    gains = {
        name: improvement_over_naive(metrics, naive_metrics)
        for name, metrics in results.items()
        if name != "Tomorrow = today"
    }
    print(pd.DataFrame.from_dict(gains, orient="index").to_string(float_format="%.3f"))

    print("\nMean absolute error by participant")
    errors = pd.DataFrame(predictions, index=shared_test.index).sub(actual, axis=0).abs()
    print(errors.groupby(shared_test["user_id"]).mean().to_string(float_format="%.3f"))

    print("\nLargest error rows (Linear / logistic)")
    print(worst_error_rows(shared_test, predictions["Linear / logistic"], top_n=5).to_string(index=False))

    if ablation:
        print("\nFeature group ablation (LinearRegression test MAE)")
        active_groups = []
        ablation_rows = []
        for group_name in FEATURE_GROUPS:
            active_groups.append(group_name)
            cols = feature_columns(train, groups=active_groups)
            if not cols:
                continue
            reg, _ = model_pairs()["Linear / logistic"]
            reg.fit(train[cols], train["target_readiness"])
            preds = np.clip(reg.predict(shared_test[cols]), 0.0, 10.0)
            mae = float(np.mean(np.abs(actual - preds)))
            ablation_rows.append({
                "step": f"+{group_name}",
                "n_features": len(cols),
                "test_mae": mae,
                "mae_gain_vs_naive": float(naive_metrics["mae"] - mae),
            })
        if ablation_rows:
            print(pd.DataFrame(ablation_rows).to_string(index=False, float_format="%.3f"))


def main() -> None:
    """Parse CLI arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/pmdata")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["time", "users"], default="time")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    run_benchmark(
        data_dir=args.data_dir,
        test_fraction=args.test_fraction,
        split_mode=args.split_mode,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()