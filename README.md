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
