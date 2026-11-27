"""Interactive estimate of tomorrow's readiness and training intensity guidance."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import load_daily_table
from src.features import READY_THRESHOLD, build_feature_table
from src.models import gradient_boosting_regressor
from src.split import time_based_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pmdata"
DEMO_FEATURES = ["readiness", "sleep_duration_h", "sleep_quality", "resting_hr", "steps"]
LABELS = {
    "readiness": "Today's readiness",
    "sleep_duration_h": "Sleep duration",
    "sleep_quality": "Sleep quality",
    "resting_hr": "Resting heart rate",
    "steps": "Daily steps",
}

PAGE_STYLE = """
<style>
.stApp {
    background: #f6f8fc;
    color: #0f172a;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
    visibility: hidden;
}
[data-testid="stDecoration"] {
    display: none;
}
.eyebrow {
    margin: 0 0 0.3rem;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1d4ed8;
}
.app-title {
    margin: 0;
    font-size: 2.15rem;
    line-height: 1.15;
    letter-spacing: -0.025em;
    font-weight: 700;
    color: #0f172a;
}
.lede {
    margin: 0.55rem 0 0.25rem;
    max-width: 48rem;
    color: #475569;
    font-size: 0.98rem;
    line-height: 1.5;
}
.panel-kicker {
    margin: 0 0 0.75rem;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 0.45rem 0.55rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
}
.score-row {
    display: flex;
    gap: 1.25rem;
    align-items: center;
    flex-wrap: wrap;
}
.ring {
    width: 132px;
    height: 132px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    background: conic-gradient(var(--tone) calc(var(--pct) * 1%), #e2e8f0 0);
}
.ring-hole {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: #ffffff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.ring-score {
    font-size: 1.85rem;
    line-height: 1.05;
    letter-spacing: -0.03em;
    font-weight: 700;
    color: #0f172a;
}
.ring-scale {
    margin-top: 0.15rem;
    color: #64748b;
    font-size: 0.74rem;
    font-weight: 500;
}
.zone-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.zone-badge.ready {
    background: #dcfce7;
    color: #166534;
}
.zone-badge.caution {
    background: #fef3c7;
    color: #92400e;
}
.zone-badge.recovery {
    background: #fee2e2;
    color: #991b1b;
}
.source-tag {
    display: inline-block;
    margin-left: 0.4rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    background: #eff6ff;
    color: #1e40af;
}
.guidance-title {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
}
.guidance-copy {
    margin: 0.3rem 0 0;
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.45;
}
.summary-box {
    margin: 0.95rem 0;
    padding: 0.85rem 1rem;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #1e293b;
    font-size: 0.92rem;
    line-height: 1.5;
}
.driver {
    margin-top: 0.75rem;
}
.driver-top {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    font-size: 0.88rem;
    font-weight: 500;
    color: #1e293b;
}
.driver-delta {
    color: #64748b;
    font-variant-numeric: tabular-nums;
    font-size: 0.84rem;
}
.track {
    margin-top: 0.32rem;
    height: 8px;
    border-radius: 999px;
    background: #e2e8f0;
    overflow: hidden;
}
.fill {
    height: 100%;
    border-radius: inherit;
}
.fill.up {
    background: #16a34a;
}
.fill.down {
    background: #dc2626;
}
.fill.flat {
    background: #94a3b8;
}
.muted-note {
    margin: 0.45rem 0 0;
    color: #64748b;
    font-size: 0.85rem;
    line-height: 1.45;
}
</style>
"""


def participant_signature(data_dir):
    """Return sorted participant folder names to key cached model training."""
    root = Path(data_dir)
    if not root.exists():
        return "missing"
    names = sorted(path.name for path in root.glob("p[0-9][0-9]") if path.is_dir())
    return ",".join(names)


def _built_in_demo_frame(n_rows: int = 360) -> pd.DataFrame:
    """Create a classical-ML demo training table when local PMData is not yet placed."""
    rng = np.random.default_rng(42)
    readiness = rng.integers(2, 10, size=n_rows).astype(float)
    sleep_hours = np.clip(rng.normal(7.2, 1.1, size=n_rows), 4.0, 10.5)
    sleep_quality = rng.integers(1, 6, size=n_rows).astype(float)
    resting_hr = np.clip(rng.normal(61.0, 6.0, size=n_rows), 45.0, 88.0)
    steps = np.clip(rng.normal(8500.0, 3200.0, size=n_rows), 1500.0, 22000.0)

    next_score = (
        0.55 * readiness
        + 0.35 * (sleep_hours - 7.0)
        + 0.35 * (sleep_quality - 3.0)
        - 0.04 * (resting_hr - 60.0)
        - 0.00008 * (steps - 8000.0)
        + 2.8
        + rng.normal(0.0, 0.65, size=n_rows)
    )
    frame = pd.DataFrame({
        "readiness": readiness,
        "sleep_duration_h": sleep_hours,
        "sleep_quality": sleep_quality,
        "resting_hr": resting_hr,
        "steps": steps,
        "target_readiness": np.clip(next_score, 0.0, 10.0),
    })
    return frame


@st.cache_resource(show_spinner="Preparing readiness model...")
def train_demo_model(data_dir, signature):
    """Fit the classical gradient-boosting regressor on local PMData or demo data."""
    name, model = gradient_boosting_regressor()

    if signature not in {"", "missing"}:
        examples = build_feature_table(load_daily_table(data_dir))
        train, test = time_based_split(examples)
        model.fit(train[DEMO_FEATURES], train["target_readiness"])
        predicted = pd.Series(model.predict(test[DEMO_FEATURES]), index=test.index).clip(0, 10)
        return {
            "name": name,
            "model": model,
            "source_label": "Local PMData",
            "is_real_data": True,
            "medians": train[DEMO_FEATURES].median(),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_people": int(train["user_id"].nunique()),
            "mae": float((predicted - test["target_readiness"]).abs().mean()),
        }

    demo_df = _built_in_demo_frame()
    split_idx = int(len(demo_df) * 0.8)
    train_df = demo_df.iloc[:split_idx]
    test_df = demo_df.iloc[split_idx:]
    model.fit(train_df[DEMO_FEATURES], train_df["target_readiness"])
    predicted = pd.Series(model.predict(test_df[DEMO_FEATURES]), index=test_df.index).clip(0, 10)
    return {
        "name": name,
        "model": model,
        "source_label": "Built-in demo data",
        "is_real_data": False,
        "medians": train_df[DEMO_FEATURES].median(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_people": 12,
        "mae": float((predicted - test_df["target_readiness"]).abs().mean()),
    }


def clipped_score(model, readings):
    """Predict a single readiness estimate clipped to the 0-10 scale."""
    predicted = pd.Series(model.predict(readings)).clip(0, 10)
    return float(predicted.iloc[0])


def input_effects(bundle, readings, score):
    """Estimate how each input shifts the score vs. a typical training day."""
    effects = []
    for column in DEMO_FEATURES:
        typical = bundle["medians"][column]
        if pd.isna(typical):
            continue
        altered = readings.copy()
        altered[column] = typical
        effects.append((column, score - clipped_score(bundle["model"], altered)))
    effects.sort(key=lambda item: abs(item[1]), reverse=True)
    return effects


def readiness_zone(score):
    """Return zone label, CSS class, and session recommendation for a 0-10 score."""
    if score >= READY_THRESHOLD:
        return "Green Zone", "ready", "Ready for high intensity"
    if score >= 5.0:
        return "Yellow Zone", "caution", "Prefer recovery / light session"
    return "Red Zone", "recovery", "Prefer recovery / light session"


def guidance_message(score, today_score):
    """Write plain-English training advice for tomorrow."""
    _, _, action = readiness_zone(score)
    diff = score - today_score
    if abs(diff) < 0.2:
        trend = "similar to today"
    elif diff > 0:
        trend = f"{diff:.1f} points higher than today"
    else:
        trend = f"{abs(diff):.1f} points lower than today"

    if score >= READY_THRESHOLD:
        return (
            f"{action}: predicted readiness is {score:.1f}/10 ({trend}). "
            "Good day for a planned heavy lift or hard conditioning session."
        )
    return (
        f"{action}: predicted readiness is {score:.1f}/10 ({trend}). "
        "Keep effort controlled with mobility, easy cardio, or rest."
    )


def score_card_html(score, source_label):
    """Render the score ring, zone pill, data-source tag, and session label."""
    zone_label, css_class, action_label = readiness_zone(score)
    tones = {"ready": "#16a34a", "caution": "#d97706", "recovery": "#dc2626"}
    pct = max(0.0, min(100.0, score / 10.0 * 100.0))
    return f"""
    <div class="score-row">
      <div class="ring" style="--pct:{pct:.1f}; --tone:{tones[css_class]};">
        <div class="ring-hole">
          <div class="ring-score">{score:.1f}</div>
          <div class="ring-scale">out of 10</div>
        </div>
      </div>
      <div>
        <div>
          <span class="zone-badge {css_class}">{zone_label}</span>
          <span class="source-tag">{source_label}</span>
        </div>
        <p class="guidance-title">{action_label}</p>
        <p class="guidance-copy">
          Scale is 0 to 10. Scores of {READY_THRESHOLD}/10 or higher support high-intensity training.
        </p>
      </div>
    </div>
    """


def factors_html(effects):
    """Render the top 3 factors driving tomorrow's score vs. typical training days."""
    top_three = effects[:3]
    peak = max((abs(val) for _, val in top_three), default=0.0)
    if peak < 0.05:
        return "<p class='muted-note'>Current inputs are close to typical training values.</p>"

    items = []
    for col, delta in top_three:
        if delta > 0.05:
            css = "up"
            label_delta = f"+{delta:.1f} vs typical"
        elif delta < -0.05:
            css = "down"
            label_delta = f"{delta:.1f} vs typical"
        else:
            css = "flat"
            label_delta = "0.0 vs typical"
        width = 10 if css == "flat" else max(12.0, abs(delta) / peak * 100.0)
        items.append(
            "<div class='driver'>"
            f"<div class='driver-top'><span>{LABELS[col]}</span>"
            f"<span class='driver-delta'>{label_delta}</span></div>"
            f"<div class='track'><div class='fill {css}' style='width:{width:.0f}%;'></div></div>"
            "</div>"
        )
    return "".join(items)


def _apply_preset(preset_name):
    """Update active inputs and prediction snapshot for a quick scenario."""
    presets = {
        "Fresh / well-rested": {
            "readiness": 8,
            "sleep_hours": 8.0,
            "sleep_quality": 4,
            "resting_hr": 56,
            "steps": 7500,
        },
        "Tired / poor sleep": {
            "readiness": 5,
            "sleep_hours": 5.5,
            "sleep_quality": 2,
            "resting_hr": 68,
            "steps": 13500,
        },
    }
    chosen = presets[preset_name]
    for key, val in chosen.items():
        st.session_state[key] = val
    st.session_state["submitted_inputs"] = dict(chosen)


def main():
    st.set_page_config(page_title="Recovery Readiness", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    st.markdown(
        """
        <p class="eyebrow">Daily Recovery Readiness</p>
        <h1 class="app-title">Should you train hard tomorrow or recover?</h1>
        <p class="lede">
          Set today's inputs and click <strong>Predict Tomorrow's Score</strong> to estimate
          tomorrow's readiness score and recommended training intensity.
        </p>
        """,
        unsafe_allow_html=True,
    )

    defaults = {
        "readiness": 7,
        "sleep_hours": 7.0,
        "sleep_quality": 3,
        "resting_hr": 60,
        "steps": 8000,
    }
    for key, default_val in defaults.items():
        st.session_state.setdefault(key, default_val)
    st.session_state.setdefault("submitted_inputs", dict(defaults))

    signature = participant_signature(DATA_DIR)
    bundle = None
    training_error = None
    try:
        bundle = train_demo_model(str(DATA_DIR), signature)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        training_error = str(exc)

    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("<p class='panel-kicker'>Today's Metrics</p>", unsafe_allow_html=True)

            p1, p2 = st.columns(2)
            p1.button(
                "Preset: Fresh day",
                use_container_width=True,
                on_click=_apply_preset,
                args=("Fresh / well-rested",),
            )
            p2.button(
                "Preset: Fatigued day",
                use_container_width=True,
                on_click=_apply_preset,
                args=("Tired / poor sleep",),
            )

            with st.form("readiness_form"):
                readiness = st.slider(
                    "Today's readiness (0–10)",
                    min_value=0,
                    max_value=10,
                    key="readiness",
                    help="0 = not ready at all, 10 = fully ready.",
                )

                s_col, a_col = st.columns(2)
                with s_col:
                    sleep_hours = st.slider(
                        "Sleep last night (hours)",
                        min_value=0.0,
                        max_value=14.0,
                        step=0.5,
                        key="sleep_hours",
                    )
                    sleep_quality = st.slider(
                        "Sleep quality (1–5)",
                        min_value=1,
                        max_value=5,
                        key="sleep_quality",
                        help="1 = poor, 3 = normal, 5 = excellent.",
                    )
                with a_col:
                    resting_hr = st.number_input(
                        "Resting heart rate (bpm)",
                        min_value=30,
                        max_value=140,
                        key="resting_hr",
                    )
                    steps = st.number_input(
                        "Steps today",
                        min_value=0,
                        max_value=40000,
                        step=500,
                        key="steps",
                    )

                predict_clicked = st.form_submit_button(
                    "Predict Tomorrow's Score",
                    type="primary",
                    use_container_width=True,
                )

            if predict_clicked:
                st.session_state["submitted_inputs"] = {
                    "readiness": readiness,
                    "sleep_hours": sleep_hours,
                    "sleep_quality": sleep_quality,
                    "resting_hr": resting_hr,
                    "steps": steps,
                }

    active = st.session_state["submitted_inputs"]
    readings = pd.DataFrame(
        [[
            active["readiness"],
            active["sleep_hours"],
            active["sleep_quality"],
            active["resting_hr"],
            active["steps"],
        ]],
        columns=DEMO_FEATURES,
    )

    if training_error is not None or bundle is None:
        score = float(active["readiness"])
        effects = []
        source_label = "Baseline fallback"
    else:
        score = clipped_score(bundle["model"], readings)
        effects = input_effects(bundle, readings, score)
        source_label = bundle["source_label"]

    with right_col:
        with st.container(border=True):
            st.markdown("<p class='panel-kicker'>Tomorrow's Outlook</p>", unsafe_allow_html=True)
            st.markdown(score_card_html(score, source_label), unsafe_allow_html=True)
            st.markdown(
                f"<div class='summary-box'>{guidance_message(score, float(active['readiness']))}</div>",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Tomorrow", f"{score:.1f} / 10")
            m2.metric("Today", f"{float(active['readiness']):.1f} / 10")
            m3.metric("Session", "High intensity" if score >= READY_THRESHOLD else "Light / recovery")

            if training_error is not None:
                st.caption(f"Local PMData note: {training_error}")

            if effects:
                st.markdown(
                    "<p class='panel-kicker' style='margin-top:1.1rem;'>Top 3 Factors vs. Typical Day</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(factors_html(effects), unsafe_allow_html=True)

            with st.expander("How this estimate works"):
                if bundle is None:
                    st.write(
                        "Showing today's score as a baseline reference while local dataset files are unavailable."
                    )
                elif bundle["is_real_data"]:
                    st.write(
                        f"Trained using {bundle['name']} on local PMData ({bundle['n_train']:,} earlier training "
                        f"rows across {bundle['n_people']} participants; held-out test MAE on {bundle['n_test']:,} "
                        f"later days: {bundle['mae']:.2f} points on a 0–10 scale)."
                    )
                else:
                    st.write(
                        f"Local PMData was not found under data/raw/pmdata, so {bundle['name']} was fit on a "
                        f"built-in classical-ML demo set ({bundle['n_train']:,} train rows, holdout MAE "
                        f"{bundle['mae']:.2f}) so you can test predictions immediately. Place real PMData "
                        "participant folders under data/raw/pmdata for actual PMData training."
                    )

                st.write(
                    "Each factor bar shows how much the predicted score shifts when that single input is "
                    "replaced by its training-set median while holding the other four inputs constant."
                )


if __name__ == "__main__":
    main()