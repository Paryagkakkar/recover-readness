# Project Journal

## 2026-10-19 — Paryag
- **Did:** Repo + folder structure, .gitignore, pinned requirements.
- **Why:** Fix structure early so each of us knows where our code goes.
- **Issue:** Notebooks created as empty files; Jupyter can't open them.
- **Next:** Pick dataset, set up environment.

## 2026-10-20 — Dev
- **Did:** Chose PMData (16 people, ~5 months, Fitbit + daily wellness). Set up Python 3.11 venv.
- **Why:** Kaggle Whoop-style datasets look synthetic ; PMData is real and used in readiness research.
- **Issues:** Streamlit needed pyarrow pinned to 17.0.0; LightGBM on Mac needs `brew install libomp`; .gitignore missed .venv/.
- **Decision:** Target is now self-reported readiness (1–10), not a 0–100 score.
- **Next:** Download PMData, inspect files.