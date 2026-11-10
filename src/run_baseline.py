"""Phase 1 results: naive baseline vs linear / logistic regression.

Run from the repo root:  python -m src.run_baseline [path/to/pmdata]

Both models are scored on the SAME test rows (the later 20% of each user's
days). The naive baseline needs today's readiness, which is missing on some
days, so the headline table only uses test rows where it exists. The linear
model is also reported on all test rows for reference.
"""

import sys

import pandas as pd

from src.data import load_daily_table
from src.evaluate import classification_metrics, mae_improvement, regression_metrics
from src.features import build_feature_table
from src.models import (build_linear_pipeline, build_logistic_pipeline,
                        feature_columns, naive_label, naive_predict)
from src.split import assert_no_future_in_train, time_based_split


def run(pmdata_dir="data/raw/pmdata"):
    table = build_feature_table(load_daily_table(pmdata_dir))
    train, test = time_based_split(table)
    assert_no_future_in_train(train, test)

    cols = feature_columns(table)
    print(f"rows: train={len(train)} test={len(test)} | features={len(cols)}")
    print(f"'ready' share: train={train['ready_high_intensity'].mean():.2f} "
          f"test={test['ready_high_intensity'].mean():.2f}")

    # Fit on train only.
    lin = build_linear_pipeline().fit(train[cols], train["target_readiness"])
    log = build_logistic_pipeline().fit(train[cols], train["ready_high_intensity"])

    # Same-rows comparison: keep test days where today's readiness is known.
    same = test[test["readiness"].notna()]
    y = same["target_readiness"]
    rows = {
        "naive (tomorrow = today)": regression_metrics(y, naive_predict(same)),
        "linear regression": regression_metrics(y, lin.predict(same[cols])),
    }
    table_reg = pd.DataFrame(rows).T[["mae", "rmse", "n"]]
    table_reg["mae_vs_naive"] = [
        0.0, mae_improvement(rows["linear regression"]["mae"],
                             rows["naive (tomorrow = today)"]["mae"])]
    print("\nRegression, test rows with today's readiness known:")
    print(table_reg.round(3).to_string())

    all_lin = regression_metrics(test["target_readiness"], lin.predict(test[cols]))
    print(f"\nLinear on ALL test rows: MAE={all_lin['mae']:.3f} "
          f"RMSE={all_lin['rmse']:.3f} n={all_lin['n']}")

    yc = same["ready_high_intensity"]
    naive_c = classification_metrics(yc, naive_label(same))
    log_c = classification_metrics(yc, log.predict(same[cols]),
                                   log.predict_proba(same[cols])[:, 1])
    print("\nClassification (ready = readiness >= 7), same rows:")
    for name, m in [("naive", naive_c), ("logistic", log_c)]:
        print(f"  {name:9s} F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f} "
              f"confusion[[TN,FP],[FN,TP]]={m['confusion']}")
    return table_reg, naive_c, log_c


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/pmdata")
