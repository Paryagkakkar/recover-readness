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

## 2026-11-06 — Team
- **Did:** Implemented `src/features.py`: next-day readiness target (`groupby(user_id).shift(-1)`), derived binary label (`ready_high_intensity` = target ≥ 7), lag-1 features for wellness/Fitbit columns, and day-of-week. Added `tests/test_leakage.py` (toy 2-user calendar) so we catch cross-user shift bugs without needing the 1.4 GB zip. Wired a short check into `01_eda.ipynb`.
- **Why:** Phase 1 needs the modelling table before splits/baselines. Leakage rules 1–2 are the highest-risk part; better to lock them with tests early.
- **Result (numbers):** Not measured on real PMData this session (local `data/raw/pmdata` not present). Toy tests cover target alignment and “lag must not cross users”.
- **Problems / what didn't work:** Current `venv` is Python 3.14; pinned `pandas==2.2.0` will not install cleanly on 3.14 (needs build from source). Journal from 2026-10-20 said 3.11 — we should recreate the venv on 3.11 before training.
- **Considered but rejected:** Rolling 3/7-day means and sleep debt in this same change — Phase 2; adding them now would mix feature groups and break the ablation plan. Also rejected putting the binary label in a separate “independent” story: it is derived from the score and the report must say so.
- **Next:** Recreate Python 3.11 venv + download PMData if missing → time-based split (`split.py`) → naive baseline + linear regression MAE.