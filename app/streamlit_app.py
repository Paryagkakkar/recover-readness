"""A small demo of tomorrow's self-reported readiness."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_daily_table
from src.features import build_feature_table
from src.models import model_pairs
from src.split import time_based_split


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pmdata"
DEMO_FEATURES = ["readiness", "sleep_duration_h", "sleep_quality", "resting_hr", "steps"]


@st.cache_resource(show_spinner="Training the demo model on local PMData…")
def train_demo_model(data_dir):
    examples = build_feature_table(load_daily_table(data_dir))
    train, _ = time_based_split(examples)
    model, _ = model_pairs()["Linear / logistic"]
    return model.fit(train[DEMO_FEATURES], train["target_readiness"])


st.set_page_config(page_title="Recovery Readiness", layout="centered")
st.title("Tomorrow’s readiness")
st.caption("A course demo that estimates tomorrow’s self-reported readiness on a 0–10 scale.")

with st.form("readiness_inputs"):
    readiness = st.number_input("Today’s readiness (0–10)", 0, 10, 7)
    sleep, activity = st.columns(2)
    with sleep:
        sleep_hours = st.number_input("Sleep last night (hours)", 0.0, 24.0, 7.0, 0.5)
        sleep_quality = st.slider("Sleep quality", 1, 5, 3, help="1 is poor and 5 is excellent.")
    with activity:
        resting_hr = st.number_input("Resting heart rate (bpm)", 20, 220, 60)
        steps = st.number_input("Steps today", min_value=0, value=8000, step=100)
    submitted = st.form_submit_button("Estimate readiness", type="primary")

if submitted:
    try:
        model = train_demo_model(DATA_DIR)
    except FileNotFoundError:
        st.metric("Simple estimate", f"{readiness:.1f} / 10")
        st.caption("The simple estimate assumes tomorrow will feel like today.")
        st.info("To see the model estimate, add PMData locally using the README instructions.")
    else:
        readings = pd.DataFrame(
            [[readiness, sleep_hours, sleep_quality, resting_hr, steps]],
            columns=DEMO_FEATURES,
        )
        score = model.predict(readings).clip(0, 10)[0]
        simple, fitted = st.columns(2)
        simple.metric("Simple estimate", f"{readiness:.1f} / 10")
        simple.caption("Tomorrow will feel like today.")
        fitted.metric("Model estimate", f"{score:.1f} / 10")
        fitted.caption("A linear model trained on the five inputs shown above.")
        st.caption("The project benchmark uses more inputs and reports results on held-out days.")
