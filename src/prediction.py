"""Five-input readiness estimate shared by the dashboard and Streamlit demo."""

from pathlib import Path

import pandas as pd

from src.data import load_daily_table
from src.features import build_feature_table
from src.models import model_pairs
from src.split import time_based_split


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pmdata"
DEMO_FEATURES = ("readiness", "sleep_duration_h", "sleep_quality", "resting_hr", "steps")


def train_demo_model(data_dir):
    """Fit the demo pipeline on the training portion of PMData."""
    examples = build_feature_table(load_daily_table(data_dir))
    train, _ = time_based_split(examples)
    model, _ = model_pairs()["Linear / logistic"]
    return model.fit(train[list(DEMO_FEATURES)], train["target_readiness"])


def estimate_readiness(readings, model):
    """Return today's score and the optional model estimate on the 0–10 scale."""
    predicted = None
    if model is not None:
        row = pd.DataFrame([readings], columns=DEMO_FEATURES)
        predicted = float(model.predict(row).clip(0, 10)[0])
    return {"simple_estimate": readings["readiness"], "model_estimate": predicted}
