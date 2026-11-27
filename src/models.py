"""Models for the readiness benchmark and the interactive demo."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_GROUPS, LAG_COLUMNS

EXCLUDED_FEATURE_COLS = {
    "user_id",
    "date",
    "target_readiness",
    "ready_high_intensity",
}


def feature_columns(table, groups=None):
    """Select feature columns available up to the prediction day."""
    if groups is None:
        candidates = ["day_of_week", *LAG_COLUMNS]
        candidates.extend(f"{column}_lag1" for column in LAG_COLUMNS)
    else:
        unknown = [g for g in groups if g not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(f"Unknown feature groups: {unknown}")
        candidates = []
        for group_name in groups:
            candidates.extend(FEATURE_GROUPS[group_name])

    seen = set()
    selected = []
    for column in candidates:
        if column in table.columns and column not in EXCLUDED_FEATURE_COLS and column not in seen:
            seen.add(column)
            selected.append(column)
    return selected


def gradient_boosting_regressor():
    """Return a fresh gradient-boosting regressor with median imputation.

    Uses XGBRegressor when installed and falls back to HistGradientBoostingRegressor.
    """
    imputer = SimpleImputer(strategy="median")
    try:
        from xgboost import XGBRegressor

        booster = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
        return "XGBoost", make_pipeline(imputer, booster)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        booster = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=3,
            learning_rate=0.08,
            min_samples_leaf=20,
            random_state=42,
        )
        return "Gradient boosting", make_pipeline(imputer, booster)


def model_pairs():
    """Return fresh score (regressor) and label (classifier) pipelines."""
    return {
        "Linear / logistic": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LinearRegression(),
            ),
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(max_iter=1000),
            ),
        ),
        "Random forest": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RandomForestRegressor(min_samples_leaf=5, random_state=42),
            ),
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RandomForestClassifier(min_samples_leaf=5, random_state=42),
            ),
        ),
    }