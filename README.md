# Contact-Centre Forecasting and Staffing: A Halifax 311 Case Study

## Project overview

Does the most accurate call-volume forecast also lead to the best staffing decisions? This project looks at eight forecasting methods and evaluates them by the staffing decisions they produce, as well as by their forecast errors.

The analysis uses half-hourly 311 call volumes published by Halifax Regional Municipality (HRM). Every Monday, each model predicts demand for the week beginning seven days later. This matches the lead time used to prepare staff schedules. The forecasts are passed through an Erlang A queueing model to estimate the number of agents needed in each half-hour. Those estimates are then compared with the staffing requirement implied by actual demand.

The test covers 52 weeks, from September 2025 through August 2026.

## Terms used in this report

- **WAPE:** Total absolute forecast error divided by the total number of actual calls.
- **Staffing error:** The difference between the productive agent-hours requested by a forecast and the productive agent-hours implied by actual demand, measured per week. One agent working for half an hour equals 0.5 agent-hours. This is an error measure, not the centre's actual working hours.
- **Cost ratio (`r`):** The relative cost of being short one agent-hour compared with having one spare agent-hour. At `r = 1`, both have the same cost.
- **Primary and secondary models:** Primary models use data from after the July 2024 platform change. Secondary models use the full series going back to 2017.
- **Scenarios:** Sensitivity tests that change one staffing assumption at a time, including the service target, caller patience, handling time, and using Erlang C instead of Erlang A.

## Main findings

### 1. The published data contained major errors

For one year, the published data overstated demand by about 61%. From May 2023 to July 2024, most days were counted twice. Starting in July 2024, timestamps also carry a fixed UTC+4 offset. HRM's documentation does not explain either issue.

These problems matter because a model trained on the published series would learn a demand pattern that did not actually occur. Both issues were corrected before training or evaluating the models. The evidence and repair process are described in the [Data](#data) section.

![Daily agent demand, as published and repaired](results/figures/repair.png)

### 2. On this data, with equal costs, WAPE ranks the models as staffing error does

When staffing is based directly on point forecasts and understaffing and overstaffing have equal costs, the WAPE ranking closely matches the ranking by staffing error. Across all six staffing scenarios, the rank correlation is between 0.95 and 1.00.

XGBoost performs best. Every other model has a higher staffing error by an amount the bootstrap separates from zero, based on paired weekly differences and a moving-block bootstrap with four-week blocks.

There is also a limit to how accurate any model can be. With roughly 32 calls per open half-hour, even a forecast that knew the true arrival rate would have an estimated WAPE of 13.4% and a staffing error of 44 agent-hours per week if calls followed a Poisson process. The best model reaches 19.3% WAPE and 63 agent-hours of weekly staffing error. The models differ only in the error left above that floor.

![WAPE against agent-hours error](results/figures/accuracy_staffing.png)

### 3. Forecast bias matters when understaffing is more expensive

WAPE measures the size of forecast errors, but it does not show whether a model tends to predict too many or too few calls. That distinction becomes important when understaffing costs more than overstaffing.

Chronos-2 and the secondary Prophet model have the same weekly staffing error, but Chronos-2 under-forecasts demand by 4.4%. It leaves 37% of intervals understaffed, compared with 29% for Prophet. Under the Erlang A model, staffing from Chronos-2 is expected to answer 78.4% of calls within 25 seconds, below the 80% target.

When the cost of understaffing is set to 16 times the cost of having spare capacity, Chronos-2 drops from third place to seventh when staffing is based on its point forecast.

Its errors also vary with the season. Like MSTL, it responds to recent demand levels but misses changes in direction. It over-forecasts November by 23% and under-forecasts January by 21%.

![Monthly bias by model](results/figures/bias_by_month.png)

### 4. A staffing buffer can be as valuable as choosing a better model

When understaffing is more costly, adding a buffer to the forecast can substantially reduce staffing cost. The buffer is a single uplift applied to the point forecast. It is estimated from the quantile of actual demand divided by forecast demand, using forecast errors from the training data. The quantile is set to `q = r / (1 + r)`, where `r` is the cost ratio.

When one short agent-hour costs four times as much as one spare agent-hour, the buffer reduces cost by 15% to 31%, depending on the model. The lowest weekly cost falls from 153 to 119. Among non-naive models using a buffer, the difference between the best and worst model can still reach 23%.

At a cost ratio of eight, the buffer reduces cost by 35% to 51%, while model choice still accounts for a difference of about 26%. Because each model's buffer is calibrated using that model's own errors, it can also compensate for systematic bias. Chronos-2 benefits the most. Calibrating separate buffers for different slot types improves results by less than 1%.

![Weekly cost by cost ratio](results/figures/buffer.png)

These findings come from one test year, one aggregate queue of this size, a one-week forecast lead time, and a linear cost model. Results may differ for smaller queues or models with stronger bias. Across the full test year, the 80th-percentile buffer covers 76% to 82% of actual demand. From January to March, coverage falls to 51% to 81%. The February-to-August calibration period does not include those months.

## Results

| Run | WAPE | Bias | Staffing error (agent-hours/week) | Difference from best | Understaffed intervals | Modelled service level |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (secondary) | 19.3% | −0.0% | 62.6 | best | 29.6% | 80.6% |
| XGBoost (primary) | 19.8% | +1.5% | 64.5 | +1.8 (−0.2 to +4.3) | 27.9% | 81.5% |
| Prophet (secondary) | 21.3% | +0.9% | 70.2 | +7.6 (+4.3 to +12.6) | 28.6% | 80.3% |
| Chronos-2, zero-shot | 21.5% | −4.4% | 70.2 | +7.5 (+3.6 to +14.0) | 37.1% | 78.4% |
| MSTL (primary) | 21.8% | −0.0% | 71.4 | +8.8 (+5.4 to +14.6) | 32.3% | 80.2% |
| MSTL (secondary) | 21.8% | −0.1% | 71.7 | +9.0 (+5.7 to +14.8) | 32.6% | 80.2% |
| Prophet (primary) | 23.5% | −1.0% | 77.5 | +14.8 (+8.3 to +24.2) | 32.9% | 79.1% |
| Seasonal Naive | 26.1% | +0.5% | 85.5 | +22.8 (+19.3 to +28.0) | 34.8% | 79.5% |
| Poisson oracle floor | 13.4% | | 44.2 | | | |

The table reports point-forecast staffing under the primary scenario and includes open intervals only. “Difference from best” is the mean weekly difference in staffing error, with a 95% moving-block bootstrap interval. If the interval includes zero, the difference between the runs is not clear. The best run is chosen on the same test year, and the seven intervals are not corrected for multiple comparisons, so read them as a guide, not a formal test. Modelled service level is the share of actual calls that Erlang A estimates would be answered within 25 seconds. It is not an observed service level.

A warmed-up Chronos-2 forecast takes about 0.3 seconds to run. XGBoost takes 1 to 2.5 seconds, while MSTL and Prophet take 15 to 90 seconds. At this scale, runtime does not determine which model is preferable.

The figure below shows a test week whose XGBoost WAPE is the median among the test weeks.

![One test week, actual against two forecasts](results/figures/week.png)

For accuracy at peak, off-peak, and after closures, along with runtime, see [baselines.md](results/baselines.md). The [decision analysis](results/decision.md) includes bootstrap intervals, the cost-ratio comparison, buffer calibration, and scenario results.

## Data

The project uses HRM's open [311 Call Volumes service](https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/311_Call_Volumes/FeatureServer/0), which contains half-hourly records dating back to 2017. The ingestion script is `src/ingest.py`.

The target is agent demand, calculated as `HANDLED + ABANDONED` calls in each half-hour. `OFFERED` also includes calls that end in the interactive voice response system (IVR), and is about 48% higher. The centre is treated as open from 08:00 to 20:00 on weekdays and from 09:00 to 17:30 on weekends. It is closed on statutory holidays. Forecasts are evaluated and converted to staffing requirements for open intervals only.

### Data issues and corrections

| Issue | What the data shows | Evidence | Correction |
|---|---|---|---|
| Fixed UTC+4 timestamps from 2024-07-17 | Weekday peaks shift to late afternoon, and Saturday at 01:00 contains calls from Friday evening | Interpreting timestamps as UTC+4 produces a profile that matches the pre-2023 profile with a correlation of 0.9997. Shifts of ±30 minutes produce correlations of 0.96–0.97. | Convert timestamps to Halifax local time. |
| Duplicate daily counts from 2023-05-23 to 2024-07-16 | Agent demand doubles overnight, then falls by half about fourteen months later | On 395 of 407 days, every published count is even. For each of those days, published calls are 2.00 to 2.03 times the number of distinct call IDs in HRM's per-call data. | Halve counts on the affected days. Repaired totals are within 1.3% of distinct call IDs on every corrected day, and match exactly on 379 of the 395 days. See [duplicate_check.csv](results/duplicate_check.csv). |

The repaired FY2023/24 volume is within 5% of HRM's published figure. The full evidence is in [data_audit.md](data/raw/data_audit.md).

The 17 July 2024 date also marks a platform change, after which the abandonment rate is lower. For that reason, the project compares two training windows:

| Window | Dates |
|---|---|
| Primary training | 2024-07-17 to 2025-08-31 |
| Secondary training | 2017-07-03 to 2025-08-31 |
| Test | 2025-09-01 to 2026-08-31 |

## Method

### Forecasting models

Each model represents a different approach a planning team could use:

- **Seasonal Naive** provides a simple baseline that more complex models should improve on.
- **MSTL** and **Prophet** use statistical methods to separate trend and seasonal patterns.
- **XGBoost** uses machine learning with calendar and lagged-demand features.
- **Chronos-2** is a pretrained foundation model evaluated without training on this call-volume series.

XGBoost predicts each half-hour using calendar information and demand from the same slot two to five weeks earlier and 52 weeks earlier. To avoid using information that would not have been available when a schedule was made, no lag is shorter than the 14 days between the forecast origin and the end of the target week.

Chronos-2 makes zero-shot forecasts using the most recent 24 weeks, which is its maximum context window. Closed days in the historical data are replaced with the most recent open day of the same weekday. No model settings are tuned using the test period.

### Turning forecasts into staffing decisions

For each open half-hour, a forecast is converted into an integer estimate of the number of agents needed. Staffing error measures the difference between that estimate and the requirement calculated from actual demand, expressed in agent-hours.

The cost model is:

`cost = r × shortfall + excess staffing`

The cost ratio `r` is tested from 0.25 to 16. Before building the models, a preliminary staffing test found that a forecast with 10% WAPE changes the required staffing level in 39% of open half-hours. Details are in [staffing_probe.md](results/staffing_probe.md).

### Assumptions and limitations

HRM publishes call counts, not staff rosters. The staffing requirements in this report are therefore counterfactual estimates, and the cost figures should not be interpreted as HRM's actual costs. The analysis measures productive staffing needed in each half-hour. It does not model breaks, shrinkage, shift patterns, total headcount, or full-time-equivalent staffing.

The queueing and service assumptions are:

- **Erlang A, with Erlang C as a comparison.** The dataset records abandoned calls, while Erlang C assumes no callers abandon the queue.
- **Service target:** 80% of calls answered within 25 seconds, based on HRM documents from 2010 to 2014. A 75%-within-25-seconds target, used in documents from 2017 to 2019, is also tested. No more recent target was found in the published material.
- **Service time:** 237 seconds, the mean talk time in the primary training period. The analysis also tests 360 seconds. It is unclear whether hold time and after-call work are included, so 237 seconds may understate total handling time.
- **Caller patience:** 120 seconds, tested over a range from 60 to 300 seconds. The available data does not include waiting times, so patience cannot be estimated directly.
- **Half-hourly steady-state assumption:** Each half-hour is sized independently, assuming Poisson arrivals. Actual demand varies more than this assumption allows. Demand for the same slot in adjacent weeks varies about 1.9 times as much as a Poisson process would suggest, meaning the real noise floor is likely higher than the oracle estimate. The data also cannot show surges within a half-hour, so modelled service levels are likely higher than the service levels a real contact centre would record.

## Contamination audit

Chronos-2 is pretrained, so its original training data could potentially overlap with the test series.

- **Source:** HRM's ArcGIS `311_Call_Volumes` service, downloaded on 2026-09-13 at 16:13 UTC, containing 148,269 rows. `src/ingest.py` records the query and pull time with each snapshot.
- **Test period:** The test window was fixed before model development. No test data was uploaded to a training or fine-tuning service.
- **Pretraining corpus:** Chronos-2 was released in October 2025. Halifax 311 is not listed in its published training data, and most test weeks occur after the model's release.
- **Exact-series overlap:** There is no evidence that Chronos-2 saw this exact series. It has seen other call and demand datasets, which is the type of general domain knowledge a zero-shot model is intended to use.
- **Remaining uncertainty:** A published training-data description cannot prove that a public series was never included. This should therefore be understood as an external zero-shot evaluation with some uncertainty about pretraining overlap.

The other models are trained only on the training windows described above.

## Related work

Bastianin, Galeotti, and Manera (2019) evaluate daily call-arrival forecasts using both statistical accuracy and the cost of Erlang C staffing at an 80/20 service target. Their evaluations select the same best models, but the mean squared error ranking differs from the cost ranking. The cost ranking also makes a clearer distinction between the two leading models.

Ding and Koole (2022) show that, under their cost model, optimal staffing can be expressed as a forecast quantile. They also find that staffing cost is close to linear in absolute forecast error, supporting the use of WAPE when overstaffing and understaffing have equal costs.

This project extends that line of work to half-hourly demand at the one-week scheduling horizon. It uses Erlang A and compares statistical models, machine learning, and a pretrained foundation model. To our knowledge, no published study has used the Halifax dataset for this type of comparison.

## Running the project

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12.

```sh
uv sync
uv run python -m src.ingest     # Download a dated snapshot into data/raw/
uv run python -m src.clean      # Create data/processed/series.parquet
uv run python -m src.duplicates # Check duplicate repairs against per-call records
uv run python -m src.probe      # Create results/staffing_probe.md
uv run python -m src.evaluate   # Backtest all models, about 4 hours on an Apple M2
uv run python -m src.decide     # Convert forecasts to staffing decisions, about 1.5 hours on first run
uv run python -m src.report     # Generate figures in results/figures/
uv run pytest                   # Add -m 'not network' to skip the Chronos-2 download
```

Snapshots, processed data, and most results are not committed to the repository. [pull_metadata.json](data/raw/pull_metadata.json) records the query, pull time, and row count of the snapshot used here. `uv.lock` pins every package, and `src/baselines.py` pins the Chronos-2 model revision. Snapshot tests will fail until a snapshot is available. The ingestion step will not replace a snapshot created on the same UTC date.

## Repository structure

```text
src/ingest.py          Download data from ArcGIS and validate snapshots
src/clean.py           Correct timestamps and duplicates, regularize the series, define data windows
src/duplicates.py      Check the duplicate repair day by day
src/erlang.py          Erlang B, C, and A calculations and agent requirements
src/probe.py           Test how sensitive staffing is to forecast error
src/baselines.py       Seasonal Naive, MSTL, Prophet, XGBoost, and Chronos-2
src/evaluate.py        Weekly backtesting and forecast metrics
src/decide.py          Staffing decisions and cost-ratio analysis
src/report.py          Generate the report figures
data/special_days.csv  Closures, named events, and source transitions
notebooks/01_eda.ipynb Explore seasonality and changes in the data source
```
