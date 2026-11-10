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

## 2026-11-06 — paryag

- **Did:** Implemented `src/features.py`: next-day readiness target (`groupby(user_id).shift(-1)`), derived binary label (`ready_high_intensity` = target ≥ 7), lag-1 features for wellness/Fitbit columns, and day-of-week. Added `tests/test_leakage.py` (toy 2-user calendar) so we catch cross-user shift bugs without needing the 1.4 GB zip. Wired a short check into `01_eda.ipynb`.
- **Why:** Phase 1 needs the modelling table before splits/baselines. Leakage rules 1–2 are the highest-risk part; better to lock them with tests early.
- **Result (numbers):** Not measured on real PMData this session (local `data/raw/pmdata` not present). Toy tests cover target alignment and “lag must not cross users”.
- **Problems / what didn't work:** Current `venv` is Python 3.14; pinned `pandas==2.2.0` will not install cleanly on 3.14 (needs build from source). Journal from 2026-10-20 said 3.11 — we should recreate the venv on 3.11 before training.
- **Considered but rejected:** Rolling 3/7-day means and sleep debt in this same change — Phase 2; adding them now would mix feature groups and break the ablation plan. Also rejected putting the binary label in a separate “independent” story: it is derived from the score and the report must say so.
- **Next:** Recreate Python 3.11 venv + download PMData if missing → time-based split (`split.py`) → naive baseline + linear regression MAE.

## 2026-11-06 — Team (session 2)
- **Did:** Added `src/split.py` (per-user time split: earlier → train, later → test), `src/models.py` naive baseline (`tomorrow ≈ today`), and `src/evaluate.py` (MAE/RMSE + improvement vs naive). Tests in `tests/test_split.py` and `tests/test_naive_baseline.py`.
- **Why:** Need a leakage-safe split and a baseline number before any real model; a model only counts if it beats “tomorrow = today”.
- **Result (numbers):** Still no PMData on this machine — only unit tests. Split keeps max(train date) < min(test date) per user.
- **Problems / what didn't work:** Nothing broken in the toy tests. Still blocked on real MAE until PMData is downloaded.
- **Considered but rejected:** Random `train_test_split` (leaks future). Held-out-users split — save for Phase 2 comparison. Linear regression — next commit once split + naive are locked.
- **Next:** Linear / logistic regression in a Pipeline (scale on train only) and report MAE vs naive on real data.

## 2026-11-10 — Dev (draft written by AI assistant — edit in your own words)
- **Did:** Merged the time-split + naive baseline branch. Recreated the venv on Python 3.11, downloaded PMData and ran the first real results. Added linear and logistic regression pipelines (median imputer + missing flags + scaler, fitted on train only), classification metrics, `src/run_baseline.py` and `tests/test_models.py`.
- **Why:** A model only counts if it beats "tomorrow = today", and we had no real number yet.
- **Result:** Per-user time split, 1,368 train / 340 test rows; "ready" share 21% train / 19% test. On the 254 test rows where today's readiness is known: naive MAE 1.059, RMSE 1.569; linear MAE 1.050, RMSE 1.372 (MAE only 0.9% better, RMSE 12.5% better). Linear on all 340 test rows: MAE 1.106. Classification (readiness >= 7): naive F1 0.612; logistic F1 0.548, ROC-AUC 0.868.
- **Problems / what didn't work:** Linear regression barely beats the naive baseline on MAE. Logistic regression with balanced class weights over-predicts "ready" (46 false positives) and has a lower F1 than naive, though its ranking is good (AUC 0.868). One pandas FutureWarning from `pd.concat` in `load_daily_table` (empty frames), not yet looked at. The naive baseline cannot score the 86 test rows where today's report is missing.
- **Considered but rejected:** Comparing models on different rows (would flatter the model); random split; adding rolling features before knowing if lags help.
- **Next:** Tree model (LightGBM / random forest) on the same split; tune the classification threshold instead of 0.5 (not on the test set); look at per-user errors; rolling features as a separate ablation.
