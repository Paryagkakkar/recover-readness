"""Small, explainable models for the readiness benchmark."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import LAG_COLUMNS


def feature_columns(table):
    """Select only measurements available by the end of the prediction day."""
    candidates = ["day_of_week", *LAG_COLUMNS]
    candidates.extend(f"{column}_lag1" for column in LAG_COLUMNS)
    return [column for column in candidates if column in table]


def model_pairs():
    """Return fresh score and label models with their preprocessing."""
    return {
        "Linear / logistic": (
            make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), LinearRegression()),
            make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
        "Random forest": (
            make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          RandomForestRegressor(min_samples_leaf=5, random_state=42)),
            make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          RandomForestClassifier(min_samples_leaf=5, random_state=42)),
        ),
    }
