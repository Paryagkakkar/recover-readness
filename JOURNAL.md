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

## 2026-11-27 — Modeling and Streamlit demo
- **Did:** Compared the simple daily guess, linear and logistic models, and a random forest on later days from the same 16 participants. Replaced the dashboard's sample predictions with the simple estimate and a compact linear model that trains on local PMData.
- **Why:** Keep the demo easy to use and tie its inputs to the measurements in the project dataset.
- **Result:** The simple guess averaged a 1.042-point miss on 265 comparable test days. The demo showed both estimates with PMData and the simple estimate alone without it.
- **Limit:** The demo uses five inputs. The benchmark includes additional inputs, and its results apply to later days from participants already in the dataset.
