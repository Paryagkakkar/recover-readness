# Daily Recovery Readiness Prediction

Predict tomorrow's self-reported readiness to train from today's wearable and wellness data,
using classical ML only.

## Dataset: PMData

**Source:** Simula Research Laboratory — <https://datasets.simula.no/pmdata/>
(mirror: <https://osf.io/vx4bk/>)
**Paper:** Thambawita et al., *PMData: A Sports Logging Dataset*, ACM MMSys 2020,
<https://doi.org/10.1145/3339825.3394926>
**License:** CC BY 4.0 — credit the authors in the report.

16 participants (p01–p16), about 5 months each (Nov 2019 – Mar 2020), wearing a Fitbit Versa 2
and filling in daily wellness reports in the PMSys app. It is real (not synthetic) data.

### Download

The data is not in git (`data/` is gitignored). The zip is 1.4 GB.

```bash
mkdir -p data/raw && cd data/raw
curl -L -o pmdata.zip https://datasets.simula.no/downloads/pmdata.zip
# Skip the large minute-level files and food photos; we only need daily data (~90 MB unzipped)
unzip pmdata.zip -d pmdata -x '*/heart_rate.json' '*/calories.json' '*/distance.json' '*.jpg' '*.jpeg' '*.png'
```

Then load everything as one daily table:

```python
from src.data import load_daily_table
df = load_daily_table("data/raw/pmdata")   # one row per user per calendar day
```

### Files per participant

| File | Granularity | What we use |
|---|---|---|
| `pmsys/wellness.csv` | one morning report per day | **readiness (target)**, fatigue, mood, sleep_duration_h, sleep_quality, soreness, stress |
| `pmsys/srpe.csv` | per training session | daily training load = Σ(perceived exertion × minutes) |
| `pmsys/injury.csv` | per report | not used yet |
| `fitbit/resting_heart_rate.json` | daily | resting HR (0 → missing) |
| `fitbit/sleep.json` | per sleep | main sleep: minutes asleep, minutes awake, efficiency |
| `fitbit/sleep_score.csv` | per night | sleep score (0–100), deep sleep minutes, sleep resting HR, restlessness |
| `fitbit/steps.json` | per minute | summed to daily steps |
| `fitbit/{sedentary,lightly_active,moderately_active,very_active}_minutes.json` | daily | activity minutes |
| `fitbit/exercise.json`, `time_in_heart_rate_zones.json` | per session / daily | not used yet |
| `fitbit/heart_rate.json`, `calories.json`, `distance.json` | per second / minute | not used (very large) |
| `googledocs/reporting.csv` | daily | meals, weight, fluids, alcohol — not used yet |
| `participant-overview.xlsx` | per person | age, height, gender, max HR — not used yet |

Missing files: p12 has no `resting_heart_rate.json` or `lightly_active_minutes.json`;
p13 has no `resting_heart_rate.json`; p08 has no `injury.csv`; p15 has no `reporting.csv`.

### Wellness scales

- **readiness: 0–10** ("0 = not ready at all, 10 = ready for anything"). Note: 0–10, not 1–10.
- fatigue, mood, sleep_quality, soreness, stress: 1–5, where 3 = normal.
- sleep_duration_h: hours.

### What the daily table looks like

`load_daily_table()` returns **2,212 user-days**, of which **1,724 have a readiness report**
(22% of days are missing a report). Each user's rows cover every calendar day between their
first and last wellness report with no gaps, so a per-user `shift(-1)` really means "tomorrow".

Readiness differs a lot between people: per-user mean ranges from 2.8 (p05) to 7.2 (p13).
Overall distribution of the 1,747 raw reports:

| readiness | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 74 | 16 | 72 | 176 | 279 | 482 | 288 | 226 | 103 | 16 | 15 |

### Cleaning decisions (in `src/data.py`)

- PMSys timestamps are UTC → converted to Europe/Oslo before taking the date.
- Fitbit dates are already local. Sleep is dated by the wake-up morning, so it lines up
  with that morning's wellness report.
- Days with two wellness reports keep the last one (23 duplicates).
- Days with no logged training session get training load 0.

## How well can we predict tomorrow?

For each person, we trained on earlier days and saved roughly the last fifth
of their labeled days for testing. Training used 1,341 days. Of the 351 test
days, 265 had both today's and tomorrow's reported scores. Every method in
the table was compared on those same 265 days. We set aside training examples
whose next-day scores would reach into that person's test period.

| Method | Average miss ↓ | Error with bigger misses weighted more ↓ | Ready-day F1 ↑ | Ready-day ranking (ROC-AUC) ↑ |
|---|---:|---:|---:|---:|
| Simple guess (tomorrow = today) | **1.042** | 1.544 | **0.622** | 0.872 |
| Linear score model + logistic classifier | 1.046 | **1.364** | 0.494 | 0.862 |
| Random forest | 1.062 | 1.365 | 0.387 | **0.902** |

The simple guess was closest on average, missing by about one point on the
0–10 scale. The trained models reduced the penalty from larger misses.
The forest ranked ready days best, but at its default cutoff it identified
only 12 of 46 ready days. The simple guess identified 28. A “ready” day means
a self-reported score of 7 or higher. These days made up 21.3% of training days
and 17.4% of the compared test days.

Results varied by person. For example, the average miss for participant p06
fell from 2.14 points with the simple guess to 1.56 with the linear model.
For p12 it rose from 0.80 to 1.93. These results describe one test split on
the same 16 participants.

### Run the comparison

With the pinned dependencies installed and PMData extracted as described
above, run these commands from the repository root:

```bash
python -m pytest -q
python -m src.run_baseline
```

The comparison prints the summary, counts of correct and incorrect ready-day
decisions, and score error by participant.
