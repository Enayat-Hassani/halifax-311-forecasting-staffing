# Contact-Centre Forecasting and Staffing: A Halifax 311 Case Study

Does the most accurate forecast also give the best staffing? This project tests that question on public call data from Halifax 311, the city's contact centre.

Nine forecast runs from five model families were compared over 52 weeks (September 2025 to August 2026). Each forecast was converted into the number of agents needed every half-hour, using Erlang A, and compared with what actual call volume required.

**Main results**

- Accuracy (WAPE) and staffing error rank the models almost the same way.
- Models with the same error can still differ in risk. A forecast that runs low leaves the queue short more often.
- When being short costs more than being overstaffed, adjusting the forecast based on its past errors reduces cost more than switching models.
- For capacity plans 13 and 26 weeks ahead, the differences between models are small after this adjustment.

Before modelling, the published data was repaired: one year was largely counted twice, and later timestamps were four hours off.

## Terms used in this report

- **WAPE:** Total forecast error divided by total actual calls. Lower is better.
- **Net staffing requirement:** The number of agents needed on the queue, before breaks, absence and shift rules (shrinkage).
- **Staffing error:** How far the agents needed by the forecast are from the agents needed by actual calls, in agent-hours per week. One agent for one half-hour is 0.5 agent-hours.
- **Cost ratio (`r`):** How much more one short agent-hour costs than one spare agent-hour. At `r = 1`, both cost the same.
- **Primary and secondary runs:** Primary runs train only on data after the July 2024 platform change. Secondary runs train on the full history back to 2017.
- **Scenarios:** Repeat runs that change one staffing assumption at a time: the service target, caller patience, handling time, or Erlang C instead of Erlang A.

## Main findings

### 1. The published data overstated one year of call volume

From May 2023 to July 2024, most days were counted twice, which overstated that year by about 61%. From July 2024, timestamps are also four hours off (a fixed UTC+4 offset). HRM's documentation does not mention either issue.

Both were fixed before any training or testing. The [Data](#data) section shows the evidence.

![Daily calls reaching the agent queue, as published and repaired](results/figures/repair.png)

### 2. When costs are equal, accuracy and staffing error agree

If being short and being overstaffed cost the same, WAPE and staffing error rank the models almost identically. Across six scenarios, the rank correlation is 0.97 to 1.00.

XGBoost (secondary) has the lowest staffing error. It is clearly better than every non-XGBoost run, but not clearly better than XGBoost (primary).

Some error cannot be removed, because calls arrive randomly. A Poisson simulation puts this floor at 13.4% WAPE and 44 agent-hours per week. The best run reaches 19.3% WAPE and 63 agent-hours. The real floor is likely higher, because actual calls vary more than a Poisson process.

![WAPE against agent-hours error](results/figures/accuracy_staffing.png)

### 3. When shortages cost more, the direction of the error matters

WAPE shows how big the error is, not whether the forecast is too high or too low. When being short is expensive, a forecast that runs low is the bigger risk.

Chronos-2 and Prophet (secondary) have the same weekly staffing error. But Chronos-2 forecasts 4.4% too low. It leaves 37% of half-hours understaffed, against 29% for Prophet. Its modelled service level is 78.4%, below the 80% target. The same average error hides a different risk.

When one short agent-hour costs 16 times a spare one, Chronos-2 falls from third place to eighth of nine.

Chronos-2 forecasts November 23% too high and January 21% too low. It only sees the last 24 weeks of history. Adding calendar features reduces its overall bias to −2.5%, but not this November–January swing. Seasonal Naive, MSTL and Prophet (primary) show a similar swing. XGBoost (which looks at the same week last year) and Prophet (secondary, with a yearly pattern) stay within about 11% in those months. The two Prophet runs also use different training windows, so this comparison cannot say which change helped.

![Monthly bias by model](results/figures/bias_by_month.png)

### 4. Adjusting the forecast for cost helps more than changing the model

Every model produces one forecast per half-hour. For Chronos-2, this is its median (50th percentile).

To adjust it, each model first makes 26 weekly forecasts inside the training period, each using only data available at the time. For every open half-hour, actual calls are divided by forecast calls. This shows how the model usually misses.

The cost ratio `r` decides how cautious to be. HRM does not publish its costs, so ratios from 0.25 to 16 are tested. The target is the quantile `q = r / (1 + r)`: `r = 4` gives `q = 0.80`, and `r = 8` gives `q = 0.89`.

For each model and ratio, the `q`-quantile of those past ratios becomes one multiplier. The forecast is multiplied by it, and staffing is scored on the separate test year. The model is not retrained, and test data is never used to set the multiplier.

When one short agent-hour costs four times a spare one, this adjustment cuts the cost by 15% to 31%, depending on the run. The lowest weekly cost falls from 153 to 119 (in spare agent-hours). Model choice still matters: among the adjusted non-naive runs, the highest cost is 23% above the lowest.

At a ratio of eight, the adjustment cuts cost by 35% to 51%, and model choice still makes a difference of about 26%. Each run is adjusted using its own past errors, so the adjustment also corrects a model that runs low or high. Chronos-2 gains the most. Using separate multipliers for weekday, weekend, peak and off-peak slots improves results by less than 1%.

![Weekly cost by cost ratio](results/figures/buffer.png)

These results come from one test year, one queue, a one-week lead and a simple linear cost. They may differ for other queues, lead times, years or cost rules. The 80th-percentile adjustment covers 75% to 82% of open half-hours overall, but only 51% to 81% from January to March. The calibration weeks run from March to August, so they contain no January or February demand.

### 5. For capacity plans, the adjustment matters more than the model

For capacity planning, models forecast weekly volume 13 and 26 weeks ahead. Volume is turned into FTE using 237 seconds per call, 85% occupancy, 32.5% shrinkage and 37.5 paid hours per week. HRM does not publish these values. They change all estimates by the same factor, so rankings stay the same. Average weekly volume needs about 14.4 FTE at 237 seconds per call, or 21.5 at 353 seconds. This leaves out safety staffing within the day and shift inefficiency, so it is not a headcount plan ([Koole, 2022](https://doi.org/10.1287/stsy.2021.0008)).

Weekly WAPE is 9% to 12% at both leads. Simply using the same week last year comes within about two points of the best model.

At a cost ratio of eight and a 13-week lead, the adjustment cuts weekly cost by 37% to 47%. After it, the runs differ by less than one FTE per week. See [capacity.md](results/capacity.md).

## Results

| Run | WAPE | Bias | Net staffing error (agent-hours/week) | Difference from lowest | Understaffed intervals | Modelled service level |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (secondary) | 19.3% | −0.0% | 62.6 | lowest | 29.5% | 80.6% |
| XGBoost (primary) | 19.8% | +1.5% | 64.5 | +1.9 (−0.0 to +4.1) | 27.9% | 81.5% |
| Prophet (secondary) | 21.4% | +0.8% | 70.2 | +7.6 (+4.4 to +12.7) | 28.8% | 80.2% |
| Chronos-2, zero-shot | 21.5% | −4.4% | 70.2 | +7.5 (+3.6 to +14.0) | 37.1% | 78.4% |
| Chronos-2 with calendar, zero-shot | 21.7% | −2.5% | 70.4 | +7.8 (+4.2 to +14.0) | 35.0% | 79.2% |
| MSTL (primary) | 21.8% | −0.0% | 71.4 | +8.8 (+5.4 to +14.5) | 32.3% | 80.2% |
| MSTL (secondary) | 21.8% | −0.1% | 71.7 | +9.1 (+5.6 to +14.7) | 32.6% | 80.2% |
| Prophet (primary) | 23.5% | −1.0% | 77.5 | +14.8 (+8.5 to +24.2) | 32.9% | 79.1% |
| Seasonal Naive | 26.1% | +0.5% | 85.5 | +22.9 (+19.2 to +28.1) | 34.8% | 79.5% |
| Poisson noise floor | 13.4% | | 44.2 | | | |

All rows use each model's own forecast, unadjusted, on open half-hours. "Difference from lowest" is the average weekly gap to XGBoost (secondary), with a 95% block-bootstrap interval. If the interval includes zero, the two runs cannot be told apart. The intervals are not corrected for multiple comparisons. Modelled service level is calculated, not measured.

Once loaded, Chronos-2 takes about 0.3 seconds per forecast. XGBoost takes 1 to 2.5 seconds, and MSTL and Prophet take 15 to 90 seconds. At this scale, speed is not a reason to choose one model over another.

The figure below shows a typical test week: the one with the median XGBoost WAPE.

![One test week, actual against two forecasts](results/figures/week.png)

[baselines.md](results/baselines.md) has accuracy at peak, off-peak and after closures, plus run times. [decision.md](results/decision.md) has the bootstrap intervals, the cost-ratio comparison, the adjustment checks and the scenario results.

## Data

The data is HRM's [311 Call Volumes](https://data-hrm.hub.arcgis.com/datasets/HRM::311-call-volumes) dataset, used under its [Open Data Licence](https://data-hrm.hub.arcgis.com/pages/open-data-licence). It has half-hourly records back to 2017. The exact snapshot used here is committed at `data/raw/call_volumes_2026-09-13.parquet`, and `pull_metadata.json` records its source, download time and SHA-256 hash.

The forecast target is calls that reached the agent queue: `HANDLED + ABANDONED` in each half-hour. `OFFERED` is about 48% higher, because it also counts calls that ended in the automated phone menu (IVR). The centre is treated as open 08:00–20:00 on weekdays and 09:00–17:30 on weekends, and closed on statutory holidays. Only open half-hours are scored and staffed.

### Data issues and corrections

| Issue | What the data shows | Evidence | Correction |
|---|---|---|---|
| Fixed UTC+4 timestamps from 2024-07-17 | Weekday peaks move to late afternoon, and Saturday 01:00 holds Friday-evening calls | Reading timestamps as UTC+4 gives a daily profile that matches the pre-2023 one with a correlation of 0.9997. Shifting by 30 minutes either way gives only 0.96–0.97. | Convert timestamps to Halifax local time. |
| Duplicate daily counts from 2023-05-23 to 2024-07-16 | Queue volume doubles overnight, then halves about fourteen months later | On 395 of 407 days, every published count is even. On each of those days, published calls are 2.00 to 2.03 times the number of distinct call IDs in HRM's per-call data. | Halve counts on the affected days. Repaired totals are within 1.3% of distinct call IDs on every corrected day, and match exactly on 379 of the 395. See [duplicate_check.csv](results/duplicate_check.csv). |

After repair, FY2023/24 volume is within 5% of HRM's published figure. [data_audit.md](data/raw/data_audit.md) has the full evidence.

17 July 2024 is also when the phone platform changed, and abandonment is lower after it. So the project trains on two windows:

| Window | Dates |
|---|---|
| Primary training | 2024-07-17 to 2025-08-31 |
| Secondary training | 2017-07-03 to 2025-08-31 |
| Test | 2025-09-01 to 2026-08-31 |

## Method

### Forecasting models

The models range from a simple baseline to a pretrained foundation model:

- **Seasonal Naive** repeats the last week.
- **MSTL** and **Prophet** model trend plus daily, weekly or yearly patterns.
- **XGBoost** uses boosted trees with calendar features and past demand.
- **Chronos-2** is a pretrained time-series model, used as is, without training on this data.

SARIMA and Holt-Winters are not included. Chronos-2 was chosen because it is pretrained, accepts calendar features, returns quantiles and runs on a CPU. Its weights are Apache-2.0. Its median forecast is used as the point forecast; its other quantiles are not compared.

XGBoost uses calendar features and demand in the same slot 2–5 and 52 weeks earlier. No input is newer than the forecast start, which is 14 days before the end of the target week.

Chronos-2 sees the latest 24 weeks. One run uses demand only; the other adds the same calendar features as XGBoost. Closed days in its history are filled with the latest open day of the same weekday. No model is tuned on test data.

Each forecast is made on Monday at 00:00, using data up to Sunday 23:30. It covers the full week that starts seven days later, which gives schedulers one week to publish rosters.

| Model | Inputs |
|---|---|
| Seasonal Naive | The last 336 half-hours |
| MSTL | Past demand; daily and weekly patterns |
| Prophet | Past demand; weekly pattern, separate weekday and weekend daily shapes; yearly pattern in the secondary run |
| XGBoost | Half-hour slot, day of week, month; same-slot demand 2–5 and 52 weeks earlier |
| Chronos-2 | The last 24 weeks of past demand |
| Chronos-2 (calendar) | As Chronos-2, plus slot, day of week and month for the forecast week |

In the one-week benchmark, models are not told about special days. The special-day list is only used to replace closed and outage days in the training history, and closed half-hours are not scored. So models do not know about holidays, storms or the busy days after a closure. In the capacity benchmark, Prophet, XGBoost and calendar Chronos-2 are given the known number of closed days in each target week; the other runs are not. [baselines.md](results/baselines.md) has the post-closure results.

### From forecast to staffing

For each open half-hour, Erlang A turns forecast volume and the service assumptions into a whole number of agents. Staffing error is the gap between that number and the number based on actual volume, in agent-hours.

The cost is:

`cost = r × shortfall + excess staffing`

with `r` from 0.25 to 16. A synthetic test shows how sensitive this is: random errors at 10% WAPE changed the agent count in 38% to 40% of half-hours. This checks the staffing calculation, not the forecasts. See [staffing_probe.md](results/staffing_probe.md).

### Assumptions and limitations

HRM publishes call counts, not rosters. This project estimates staffing and cost; it does not reproduce what HRM actually scheduled. The one-week analysis leaves out shrinkage, shifts and roster rules. The capacity analysis converts volume to FTE but does not model hiring, attrition or training time.

The queueing and service assumptions are:

- **Erlang A, with Erlang C as a check.** Erlang A allows callers to hang up; Erlang C does not. Both assume random (Poisson) arrivals, exponential call times and a steady state within each half-hour. Erlang A also assumes exponential patience. Abandoned calls count as missing the service target.
- **Service target:** 80% of calls answered within 25 seconds, from HRM documents of 2010–2014. 75% within 25 seconds, from 2017–2019 documents, is also tested. No newer target was found.
- **Service time:** 237 seconds, the average talk time in the primary training window. HRM defines talk time as time spent speaking with an agent. Full call duration, including queue and wrap-up, averages 353 seconds; 360 seconds is also tested.
- **Caller patience:** 120 seconds, tested from 60 to 300 seconds. The data has no waiting times, so patience cannot be measured.
- **Each half-hour on its own:** Every half-hour is staffed independently. Demand in the same slot varies about 1.9 times more from week to week than a Poisson process predicts, so the simulated noise floor is too low. Half-hourly data also hides short surges, so modelled service levels may be too high.

## Chronos-2 pretraining overlap

Chronos-2 was released in October 2025. Halifax 311 is not listed in its published training data, and most test weeks come after its release. But the series is public, so overlap cannot be fully ruled out. The other models only see the training windows above.

## Related work

Bastianin, Galeotti and Manera (2019) compare daily call-arrival forecasts on both accuracy and the cost of Erlang C staffing at an 80/20 target. Both measures pick the same leading models, though the full rankings differ.

Ding and Koole (2022) study staffing in two stages, with changes during the day. Under their cost model, the best initial staffing level is a quantile of the forecast, not its mean. They also find that staffing cost grows roughly in line with forecast error ([paper](https://doi.org/10.1017/S0269964820000595)).

This project tests a simpler, data-based version of that idea (adjusting forecasts by their past errors) against half-hourly staffing cost. It does not implement their two-stage model.

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
uv run python -m src.capacity   # Create results/capacity.md, about 15 minutes
uv run pytest                   # Add -m 'not network' to skip the Chronos-2 download
```

These commands use the committed snapshot. To download a fresh one, run `uv run python -m src.ingest`; cleaning then uses the newest snapshot in `data/raw/`. Processed data and intermediate forecasts are not committed. `uv.lock` pins every package, and `src/baselines.py` pins the Chronos-2 model version.

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
