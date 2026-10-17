# Contact-Centre Forecasting and Staffing: A Halifax 311 Case Study

## Project overview

This case study measures how contact-volume forecast accuracy and bias carry into net staffing requirements. It compares eight forecasting runs from five model families, first with equal shortage and surplus costs, then with a higher penalty for understaffing.

The analysis uses half-hourly 311 call volumes published by Halifax Regional Municipality (HRM). Every Monday, each run forecasts the week beginning seven days later. Erlang A converts forecast volume and service time into the net agents required in each half-hour. That requirement is compared with the requirement calculated from actual calls.

The test covers 52 weeks, from September 2025 through August 2026.

## Terms used in this report

- **WAPE:** Total absolute forecast error divided by the total number of actual calls.
- **Net staffing requirement:** The productive agents needed in the queue before shrinkage and schedule constraints.
- **Staffing error:** The absolute difference between the net agent-hours required from forecast and actual volume, measured per week. One agent required for one half-hour equals 0.5 agent-hours.
- **Cost ratio (`r`):** The relative cost of being short one agent-hour compared with having one spare agent-hour. At `r = 1`, both have the same cost.
- **Primary and secondary runs:** Primary runs use data from after the July 2024 platform change. Secondary runs use the full series going back to 2017.
- **Scenarios:** Sensitivity tests that change one staffing assumption at a time, including the service target, caller patience, handling time, and using Erlang C instead of Erlang A.

## Main findings

### 1. The published data overstated one year of call volume

For one year, the published data overstated demand by about 61%. From May 2023 to July 2024, most days were counted twice. Starting in July 2024, timestamps also carry a fixed UTC+4 offset. HRM's documentation does not explain either issue.

Both issues were corrected before training or evaluation. The evidence and repair process are described in the [Data](#data) section.

![Daily calls reaching the agent queue, as published and repaired](results/figures/repair.png)

### 2. With equal costs, WAPE and net staffing error give almost the same ranking

When staffing is based directly on point forecasts and understaffing and overstaffing have equal costs, the WAPE ranking closely matches the ranking by staffing error. Across all six staffing scenarios, the rank correlation is between 0.95 and 1.00.

XGBoost (secondary) has the lowest net staffing error. Its bootstrap interval separates it from every non-XGBoost run, but not from XGBoost (primary), based on paired weekly differences and a moving-block bootstrap with four-week blocks.

Sampling variation limits the accuracy any model can reach. A simulation using each observed interval count as its Poisson rate produces an estimated WAPE of 13.4% and a net staffing error of 44 agent-hours per week. The lowest-error run reaches 19.3% WAPE and 63 agent-hours of weekly staffing error. This is an optimistic lower bound because actual arrivals are more variable than a Poisson process.

![WAPE against agent-hours error](results/figures/accuracy_staffing.png)

### 3. Forecast bias matters when understaffing is more expensive

WAPE measures the size of forecast errors, but it does not show whether a model tends to predict too many or too few calls. That distinction becomes important when understaffing costs more than overstaffing.

Chronos-2 and the secondary Prophet model have the same weekly staffing error, but Chronos-2 under-forecasts demand by 4.4%. It leaves 37% of intervals understaffed, compared with 29% for Prophet. Under Erlang A, its net staffing requirement gives a modelled service level of 78.4%, below the 80% target.

When the cost of understaffing is set to 16 times the cost of having spare capacity, Chronos-2 drops from third place to seventh when staffing is based on its point forecast.

The univariate Chronos-2 run over-forecasts November by 23% and under-forecasts January by 21%. It uses 24 weeks of the target series and no calendar or holiday covariates. Seasonal Naive, MSTL and Prophet (primary) show the same swing; none has an annual signal. Both XGBoost runs, which use a 52-week lag, and Prophet (secondary), with yearly seasonality, keep November and January within about 11%. The Prophet runs also use different training windows, so their gap does not isolate either change.

![Monthly bias by model](results/figures/bias_by_month.png)

### 4. A calibrated forecast uplift can matter as much as model choice

When understaffing is more costly, each point forecast is multiplied by an uplift calibrated on 26 training weeks. The uplift is the `q`-quantile of actual divided by forecast volume, where `q = r / (1 + r)` and `r` is the cost ratio.

When one short agent-hour costs four times as much as one spare agent-hour, the uplift reduces the modelled penalty by 15% to 31%, depending on the run. The lowest weekly penalty falls from 153 to 119 spare-agent-hour equivalents. Among non-naive runs using an uplift, the difference between the lowest and highest penalty is still 23%.

At a cost ratio of eight, the uplift reduces the penalty by 35% to 51%, while model choice still accounts for a difference of about 26%. Each run uses its own calibration errors, so the uplift also adjusts systematic bias. Chronos-2 benefits the most. Separate uplifts for weekday or weekend peak and off-peak slots improve the result by less than 1%.

![Weekly cost by cost ratio](results/figures/buffer.png)

These findings come from one test year, one aggregate queue, a one-week forecast lead time, and a linear penalty. Results may differ for smaller queues or runs with stronger bias. Across the full test year, the 80th-percentile uplift covers 76% to 82% of actual volume. From January to March, coverage falls to 51% to 81%. The calibration targets cover March through August, with no January or February demand.

## Results

| Run | WAPE | Bias | Net staffing error (agent-hours/week) | Difference from lowest | Understaffed intervals | Modelled service level |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (secondary) | 19.3% | −0.0% | 62.6 | lowest | 29.6% | 80.6% |
| XGBoost (primary) | 19.8% | +1.5% | 64.5 | +1.8 (−0.2 to +4.3) | 27.9% | 81.5% |
| Prophet (secondary) | 21.3% | +0.9% | 70.2 | +7.6 (+4.3 to +12.6) | 28.6% | 80.3% |
| Chronos-2, zero-shot | 21.5% | −4.4% | 70.2 | +7.5 (+3.6 to +14.0) | 37.1% | 78.4% |
| MSTL (primary) | 21.8% | −0.0% | 71.4 | +8.8 (+5.4 to +14.6) | 32.3% | 80.2% |
| MSTL (secondary) | 21.8% | −0.1% | 71.7 | +9.0 (+5.7 to +14.8) | 32.6% | 80.2% |
| Prophet (primary) | 23.5% | −1.0% | 77.5 | +14.8 (+8.3 to +24.2) | 32.9% | 79.1% |
| Seasonal Naive | 26.1% | +0.5% | 85.5 | +22.8 (+19.3 to +28.0) | 34.8% | 79.5% |
| Poisson noise floor | 13.4% | | 44.2 | | | |

The table converts each point forecast directly into a net staffing requirement under the primary scenario and includes open intervals only. “Difference from lowest” is the mean weekly difference in staffing error, with a 95% moving-block bootstrap interval. An interval containing zero does not separate the runs. The reference run is selected on the same test year, and the seven intervals are not corrected for multiple comparisons. Modelled service level is the share of actual calls that Erlang A estimates would be answered within 25 seconds; it is not an observed service level.

A warmed-up Chronos-2 forecast takes about 0.3 seconds to run. XGBoost takes 1 to 2.5 seconds, while MSTL and Prophet take 15 to 90 seconds. At this scale, runtime does not determine which model is preferable.

The figure below shows a test week whose XGBoost WAPE is the median among the test weeks.

![One test week, actual against two forecasts](results/figures/week.png)

For accuracy at peak, off-peak, and after closures, along with runtime, see [baselines.md](results/baselines.md). The [decision analysis](results/decision.md) includes bootstrap intervals, the cost-ratio comparison, uplift calibration, and scenario results.

## Data

The project uses HRM's [311 Call Volumes](https://data-hrm.hub.arcgis.com/datasets/HRM::311-call-volumes) data under its [Open Data Licence](https://data-hrm.hub.arcgis.com/pages/open-data-licence). The service contains half-hourly records dating back to 2017. The snapshot used for the reported results is committed at `data/raw/call_volumes_2026-09-13.parquet`; `pull_metadata.json` records its source, extraction time and SHA-256 hash.

The forecast target is calls that reached the agent queue, calculated as `HANDLED + ABANDONED` in each half-hour. `OFFERED` also includes calls that ended in the interactive voice response system (IVR), making it about 48% higher. The centre is treated as open from 08:00 to 20:00 on weekdays and from 09:00 to 17:30 on weekends, with statutory holidays closed. Only open intervals are evaluated and converted to net staffing requirements.

### Data issues and corrections

| Issue | What the data shows | Evidence | Correction |
|---|---|---|---|
| Fixed UTC+4 timestamps from 2024-07-17 | Weekday peaks shift to late afternoon, and Saturday at 01:00 contains calls from Friday evening | Interpreting timestamps as UTC+4 produces a profile that matches the pre-2023 profile with a correlation of 0.9997. Shifts of ±30 minutes produce correlations of 0.96–0.97. | Convert timestamps to Halifax local time. |
| Duplicate daily counts from 2023-05-23 to 2024-07-16 | Agent-queue volume doubles overnight, then falls by half about fourteen months later | On 395 of 407 days, every published count is even. For each of those days, published calls are 2.00 to 2.03 times the number of distinct call IDs in HRM's per-call data. | Halve counts on the affected days. Repaired totals are within 1.3% of distinct call IDs on every corrected day, and match exactly on 379 of the 395 days. See [duplicate_check.csv](results/duplicate_check.csv). |

The repaired FY2023/24 volume is within 5% of HRM's published figure. The full evidence is in [data_audit.md](data/raw/data_audit.md).

The 17 July 2024 date also marks a platform change, after which the abandonment rate is lower. For that reason, the project compares two training windows:

| Window | Dates |
|---|---|
| Primary training | 2024-07-17 to 2025-08-31 |
| Secondary training | 2017-07-03 to 2025-08-31 |
| Test | 2025-09-01 to 2026-08-31 |

## Method

### Forecasting models

The five model families cover a last-week baseline, statistical models, gradient-boosted trees, and a pretrained time-series model:

- **Seasonal Naive** repeats the last observed week.
- **MSTL** and **Prophet** model trend and recurring daily, weekly, or yearly patterns.
- **XGBoost** uses gradient-boosted trees with calendar and lagged-demand features.
- **Chronos-2** is a pretrained time-series model evaluated without training on this call-volume series.

XGBoost predicts each half-hour using calendar information and demand from the same slot two to five weeks earlier and 52 weeks earlier. To avoid using information that would not have been available when a schedule was made, no lag is shorter than the 14 days between the forecast origin and the end of the target week.

Chronos-2 makes univariate zero-shot forecasts from the target series using the most recent 24 weeks, which is its maximum context window. It receives no calendar or holiday covariates. Closed days in the historical data are replaced with the most recent open day of the same weekday. No model settings are tuned using the test period.

### Turning forecasts into net staffing requirements

For each open half-hour, forecast volume and the service assumptions are converted into an integer net staffing requirement. Staffing error is the absolute difference from the requirement calculated with actual volume, expressed in agent-hours.

The cost model is:

`cost = r × shortfall + excess staffing`

The cost ratio `r` is tested from 0.25 to 16. Before building the models, a preliminary staffing test found that a forecast with 10% WAPE changes the required staffing level in 39% of open half-hours. Details are in [staffing_probe.md](results/staffing_probe.md).

### Assumptions and limitations

HRM publishes call counts, not staff rosters. The analysis therefore estimates net staffing requirements, not HRM's actual staffing or costs. It does not apply shrinkage or convert interval requirements into feasible shifts, schedules, headcount, or full-time equivalents.

The queueing and service assumptions are:

- **Erlang A, with Erlang C as a comparison.** The dataset records abandoned calls, while Erlang C assumes no callers abandon the queue.
- **Service target:** 80% of calls answered within 25 seconds, based on HRM documents from 2010 to 2014. A 75%-within-25-seconds target, used in documents from 2017 to 2019, is also tested. No more recent target was found in the published material.
- **Service time:** 237 seconds, the mean talk time in the primary training period. The analysis also tests 360 seconds. It is unclear whether hold time and after-call work are included, so 237 seconds may understate total handling time.
- **Caller patience:** 120 seconds, tested over a range from 60 to 300 seconds. The available data does not include waiting times, so patience cannot be estimated directly.
- **Half-hourly steady-state assumption:** Each half-hour is sized independently, assuming Poisson arrivals. Actual demand varies more than this assumption allows. Demand for the same slot in adjacent weeks varies about 1.9 times as much as a Poisson process would suggest, meaning the real noise floor is likely higher than the simulated estimate. The data also cannot show surges within a half-hour, so modelled service levels are likely higher than the service levels a real contact centre would record.

## Chronos-2 pretraining overlap

Chronos-2 was released in October 2025. Halifax 311 is not listed in its published training data, and most test weeks occur after its release. The series is public, so its absence from the published description cannot rule out overlap. The other models use only the training windows described above.

## Related work

Bastianin, Galeotti, and Manera (2019) evaluate daily call-arrival forecasts using statistical accuracy and the cost of Erlang C staffing at an 80/20 service target. The measures select the same leading models, although the full rankings differ.

Ding and Koole (2022) show that, under their cost model, optimal staffing can be expressed as a forecast quantile. They also find that staffing cost is close to linear in absolute forecast error, supporting the use of WAPE when overstaffing and understaffing have equal costs.

This project applies the same forecast-to-staffing question to half-hourly volume at a one-week scheduling horizon. It uses Erlang A and compares statistical models, gradient-boosted trees, and a pretrained time-series model.

## Running the project

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12.

```sh
uv sync
uv run python -m src.clean      # Build the cleaned half-hourly queue volume
uv run python -m src.duplicates # Check duplicate repairs against per-call records
uv run python -m src.probe      # Create results/staffing_probe.md
uv run python -m src.evaluate   # Backtest all models, about 4 hours on an Apple M2
uv run python -m src.decide     # Calculate net staffing and penalties, about 1.5 hours on first run
uv run python -m src.report     # Generate figures in results/figures/
uv run pytest                   # Add -m 'not network' to skip the Chronos-2 download
```

The commands above use the committed snapshot. Run `uv run python -m src.ingest` to download a current dated snapshot; subsequent cleaning uses the latest snapshot in `data/raw/`. Processed data and intermediate forecasts are not committed. `uv.lock` pins every package, and `src/baselines.py` pins the Chronos-2 model revision.

## Repository structure

```text
src/ingest.py          Download data from ArcGIS and validate snapshots
src/clean.py           Correct timestamps and duplicates, regularize the series, define data windows
src/duplicates.py      Check the duplicate repair day by day
src/erlang.py          Erlang B, C, and A calculations and net staffing requirements
src/probe.py           Test how sensitive staffing is to forecast error
src/baselines.py       Seasonal Naive, MSTL, Prophet, XGBoost, and Chronos-2
src/evaluate.py        Weekly backtesting and forecast metrics
src/decide.py          Net staffing, forecast uplifts, and cost-ratio analysis
src/report.py          Generate the report figures
data/special_days.csv  Closures, named events, and source transitions
notebooks/01_eda.ipynb Explore seasonality and changes in the data source
```
