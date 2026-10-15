# WFM Forecasting Benchmark

This project tests forecasts of contact-centre demand by the staffing decision they cause, not only by their error. The data is Halifax Regional Municipality's half-hourly 311 call-volume series. Each forecast goes through an Erlang A staffing model, and the question is whether the most accurate forecast also gives the smallest staffing error.

## Status

| Step | State |
|---|---|
| Ingest and data audit | Done. `src/ingest.py`, `data/raw/data_audit.md` |
| Staffing sensitivity probe | Done. Forecast error moves the staffing requirement (see below) |
| Cleaning and train/test split | Done. `src/clean.py`, `notebooks/01_eda.ipynb` |
| Baselines and evaluation harness | Done. `src/evaluate.py`, `results/baselines.md` |
| XGBoost | Done. Same backtest, `src/baselines.py` |
| Chronos-2 | Done. Zero-shot, same backtest |
| Decision layer | Done. `src/decide.py`, `results/decision.md` |
| Report | Planned |

## The data

The target is agent demand: `HANDLED + ABANDONED` per half-hour. `OFFERED` counts every call to the line, including calls that end in the IVR, and runs about 48% higher. Erlang needs the calls that reached or queued for an agent. Outside agent hours the line still takes calls, almost all of them end in the IVR, and `ABANDONED` stays near zero, so `ABANDONED` counts callers who left the agent queue, not IVR hang-ups. HRM does not publish whether very short abandons are excluded, or how transfers and callbacks are counted.

Agents work published hours: weekdays 08:00–20:00 and weekends 09:00–17:30. Statutory holidays have been closed since 2023-05-22. Forecasts are scored and staffed only on open intervals, 24 on a weekday and 17 at a weekend.

## Data cleaning

The published series has two defects. Either one would corrupt a forecast trained on it, and neither is documented by HRM. Both were found and repaired before any model was built. [data_audit.md](data/raw/data_audit.md) has the full evidence.

| Defect | How it shows | Evidence | Repair |
|---|---|---|---|
| From 2024-07-17, each interval is a fixed UTC+4 timestamp | The weekday peak moves to the late afternoon, and Saturday 01:00 carries Friday-evening calls | Read as UTC+4, the profile matches the pre-2023 profile at correlation 0.9997, against 0.96–0.97 at ±30 minutes | Convert the labels to Halifax local time |
| From 2023-05-23 to 2024-07-16, most days are counted twice | Agent demand doubles overnight and halves fourteen months later | On 395 of 407 days every count is even. On each of those days, published calls are twice the distinct call IDs in HRM's per-call dataset | Halve the doubled days. Repaired calls then equal the distinct call IDs on every day, and the weekday level stays at 830–900 calls a day |

The repaired FY2023/24 volume matches HRM's published figure within 5%. As published, it was 61% too high. 2024-07-17 is still a break: the cutover to an Enghouse platform, after which abandonment is lower. Models train on two windows to test whether history from the old platform helps.

| Window | Dates |
|---|---|
| Primary training | 2024-07-17 → 2025-08-31 |
| Secondary training | 2017-07-03 → 2025-08-31 |
| Test | 2025-09-01 → 2026-08-31 |

## Forecast setting

Each Monday at 00:00, every model forecasts the Monday–Sunday week that starts seven days later. A schedule is built from this forecast, so it is the forecast that sets the staffing requirement. Day-ahead forecasts drive intraday changes, not the schedule. The test window has 52 such weeks.

Closed days and the 2019 outage are replaced in the history by the last open day of the same weekday before any model sees them. Without this, holiday zeros pull down the Monday profile. Targets are not changed. Model settings are fixed in advance, and none is tuned on the test window.

## Results so far

Over 52 weeks of open intervals, Seasonal Naive has a WAPE of 26.1%. MSTL has 21.8% on either window. Prophet has 23.5% on primary training and 21.3% on secondary, where it can learn the yearly cycle. Long history does not change MSTL, because its forecast follows the latest seasonal cycles. On the first open weekday after a closure, these three under-forecast by 5–9%.

XGBoost forecasts every slot directly from the calendar and from the same slot 2–5 weeks and 52 weeks earlier. No lag is shorter than the 14 days between the origin and the end of the target week. It has a WAPE of 19.8% on primary training and 19.3% on secondary, and it is the most accurate model at peak, off-peak and after closures. Chronos-2 forecasts zero-shot from the last 24 weeks, its maximum context, with no training on this data. It has a WAPE of 21.5%, close to MSTL and behind XGBoost, and it under-forecasts by 4.4%. This bias does not come from scoring the median: on training-only validation weeks the pipeline's mean output equals its median, and both over-forecast by 6.5%. Like MSTL, it follows the recent level and misses the seasonal turns: it over-forecasts November by 23% and under-forecasts January by 21%. It has no calendar input, so after a closure it under-forecasts by 12%. The model was released in October 2025, inside the test window, and Halifax 311 is not in its published training data. A warm forecast takes 0.3 s, against 1–2.5 s for XGBoost and 15–90 s for MSTL and Prophet.

[results/baselines.md](results/baselines.md) has the peak, off-peak, post-closure and run-cost tables.

## Staffing results

Each forecast is turned into the integer agent requirement per open half-hour and compared with the requirement of actual demand. The question was whether WAPE is enough to choose a forecast for staffing. On this series, it is. Staffed from the point forecast with symmetric costs, the WAPE ranking and the agent-hours error ranking of the eight runs have a rank correlation of 0.95–1.00 in all six scenarios. The XGBoost runs are best. Every non-XGBoost run is worse by an interval that excludes zero (moving-block bootstrap over the 52 weeks). The two XGBoost training windows are not separated on agent-hours error.

Much of the error cannot be removed. At about 32 calls per open half-hour, an oracle forecast that knew the true rate would still have a WAPE of 13.4% and 44 agent-hours of error a week if calls were Poisson. The best model has 19.3% and 63. Real call arrivals vary more than Poisson: same-slot demand in adjacent weeks varies about 1.9 times as much as Poisson allows, part of which is real week-to-week change. The true floor is therefore higher. The models differ only in what is left above it, and this is why their accuracy and staffing rankings agree.

WAPE hides the direction of error, and staffing shows it. Chronos-2 and Prophet on secondary training have the same agent-hours error, but Chronos-2 under-forecasts by 4.4%: 37% of its intervals are understaffed, against 29%, and its agents answer 78.4% of actual calls in 25 s, below the 80% target. When understaffing costs more and staff are set from the point forecast, the bias moves the ranking. Chronos-2 falls from third to seventh at sixteen times, XGBoost on primary training (bias +1.5%) moves to first, and the rank correlation over the five models falls from 1.00 to 0.70.

When understaffing costs more than overstaffing, the buffer matters more than the model. At four times the cost, one uplift on the point forecast lowers the lowest weekly cost from 153 to 119, a fall of 23%. The uplift is the 80th percentile of actual / forecast, from forecast errors inside training only. Setting a separate quantile for peak and off-peak, weekday and weekend slots adds less than 1%. With a buffer, the model ranking does not change with the cost ratio, and the worst run costs up to 1.6 times the best. Over the year, the 80% quantile covers 76–82% of actual demand, but only 51–81% in January to March, which the calibration weeks (February to August) do not represent.

These results hold for one aggregate queue at this volume, a one-week lead and models with small bias. A smaller queue, models with larger bias or costs that are not linear may change them.

[results/decision.md](results/decision.md) has the bootstrap, split, sweep, calibration and scenario tables, and [results/cost_ratio.png](results/cost_ratio.png) the sweep.

## Staffing probe

Before any model exists, the probe adds synthetic error of fixed size to actual demand in the primary training window and counts how often the integer staffing requirement changes. At 10% WAPE, 39% of open half-hours need a different number of agents, and the difference grows with the error. So a staffing comparison of models is worth building. This shows only that staffing responds to error size. It does not show that the accuracy and staffing rankings will differ.

## Staffing assumptions

Halifax publishes call counts, not rosters. Staffing figures are therefore counterfactual: the agents a forecast implies against the agents actual demand implies. They are not schedule accuracy.

The unit of staffing is the productive agent requirement per half-hour. Breaks, leave, training and shift patterns turn a requirement into a roster, and they are outside scope. Shrinkage is excluded from the comparison. One constant factor would scale both requirements alike, apart from integer rounding. Shrinkage that varies by interval would not, and it is not modelled. No figure is a claim about HRM costs or savings.

- **Erlang A, with Erlang C as comparator.** The data records abandoned calls, and Erlang C assumes none.
- **Service level: 25 seconds, at 80% and 75%.** HRM documents state 80% within 25 s (2010 council report, 2013/14 budget) and 75% within 25 s (2017/18–2018/19 budgets). No later document states a current target.
- **Service time is talk time, swept upward.** Mean talk time is 237 s in primary training. It drifts: 182 s in 2017 and 288 s in 2023. HRM publishes no field dictionary, so it is not known whether hold and after-call work are included. Talk time is therefore a lower bound on handle time.
- **Patience is a scenario, not an estimate.** Abandonment comes from patience and congestion together. Without waiting times or staffing levels, the data cannot separate the two, so patience is swept from 60 to 300 s.
- **Each half-hour is sized on its own.** Erlang treats an interval as steady state with Poisson arrivals. Surges inside an interval and calls that cross an interval boundary are not modelled. The model-implied service level under actual demand may be particularly sensitive to this. It is not the service level a real centre would record.
- **WAPE is the headline accuracy metric.** It weights error by volume, and under symmetric costs it approximates staffing cost (see Related work), which the decision layer tests. MAPE is not used, because it is unstable on low-volume intervals. WAPE hides the direction and timing of error, so the decision layer also reports bias, understaffed intervals, absolute agent-hours error with its short and over parts, and model-implied service level, at peak and off-peak hours separately.

## Related work

Bastianin, Galeotti and Manera (2019) evaluate daily call-arrival forecasts both statistically and by the cost of Erlang C staffing at 80/20. Both evaluations select the same best models, but the MSE ranking differs from the cost ranking, and only the cost ranking separates the two best. Ding and Koole (2022) show, under their cost model, that optimal staffing is a quantile of the forecast and that staffing cost is close to linear in absolute forecast error. With symmetric over- and understaffing costs, this supports a weighted absolute error such as WAPE. The approximation can fail for small queues. This project asks the same question on half-hourly data at the schedule horizon, with Erlang A and machine-learning and foundation models, and the decision layer tests whether WAPE predicts staffing error here. To our knowledge, no published study uses the Halifax data.

## Run it

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12.

```sh
uv sync
uv run python -m src.ingest   # pull a dated snapshot into data/raw/
uv run python -m src.clean    # write data/processed/series.parquet
uv run python -m src.duplicates # check the duplicate repair against per-call records
uv run python -m src.probe    # write results/staffing_probe.md
uv run python -m src.evaluate # backtest every model, about 4 hours on an Apple M2
uv run python -m src.decide   # staff every forecast, about 1.5 hours the first time
uv run pytest                 # add -m 'not network' to skip the Chronos-2 download
```

Snapshots, processed data and results are not committed. The snapshot tests fail until a snapshot exists. A pull refuses to replace a snapshot from the same UTC date.

## Repository

```text
src/ingest.py          ArcGIS pull and snapshot checks
src/clean.py           time-base and duplicate repair, regular series, windows
src/duplicates.py      day-by-day check of the duplicate repair
src/erlang.py          Erlang B, C and A; required agents
src/probe.py           staffing sensitivity probe
src/baselines.py       Seasonal Naive, MSTL, Prophet, XGBoost, Chronos-2
src/evaluate.py        weekly backtest and metrics
src/decide.py          staffing from forecasts, cost-ratio sweep
data/special_days.csv  closed days, named events, source transitions
notebooks/01_eda.ipynb seasonality and source breaks
```
