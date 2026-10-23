# Decision layer

52 weekly origins; open intervals only. Forecast and actual volume are converted to integer net staffing. Agent-hours are reported per week. One agent for one half-hour equals 0.5 agent-hours. These are estimates, not rosters or HRM costs.

This report tests how model choice affects staffing cost and shortage risk. It compares WAPE rankings with net staffing error and cost, including cases where understaffing costs more than spare capacity.

| Input | Value | Source |
|---|---|---|
| Service time | 237 s | Mean talk time, primary training (data) |
| Service level | 80% in 25 s | HRM documents; no current target published (assumption) |
| Patience | 120 s | Not identifiable from the data (assumption) |
| Queue model | Erlang A | Abandonment is recorded (choice) |
| Cost ratio r = c_u / c_o | 0.25 to 16 | Swept, not estimated (assumption) |

## Forecast accuracy and net staffing

Net staffing uses point forecasts and the primary scenario. Modelled service level is the share of actual calls Erlang A estimates agents answer within 25 seconds. It is not measured. The Poisson floor is optimistic because arrivals are overdispersed.

| Run | WAPE | Bias | WAPE rank | Net staffing error | Short | Over | Error rank | Understaffed | Modelled service level |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seasonal_naive (any) | 26.1% | +0.5% | 9 | 85.5 | 41.9 | 43.5 | 9 | 34.8% | 79.5% |
| mstl (primary) | 21.8% | -0.0% | 6 | 71.4 | 35.8 | 35.6 | 6 | 32.3% | 80.2% |
| mstl (secondary) | 21.8% | -0.1% | 7 | 71.7 | 36.1 | 35.6 | 7 | 32.6% | 80.2% |
| prophet (primary) | 23.5% | -1.0% | 8 | 77.5 | 40.0 | 37.5 | 8 | 32.9% | 79.1% |
| prophet (secondary) | 21.4% | +0.8% | 3 | 70.2 | 33.0 | 37.2 | 4 | 28.8% | 80.2% |
| xgboost (primary) | 19.8% | +1.5% | 2 | 64.5 | 29.7 | 34.8 | 2 | 27.9% | 81.5% |
| xgboost (secondary) | 19.3% | -0.0% | 1 | 62.6 | 31.7 | 30.9 | 1 | 29.5% | 80.6% |
| chronos2 (any) | 21.5% | -4.4% | 4 | 70.2 | 41.9 | 28.3 | 3 | 37.1% | 78.4% |
| chronos2_calendar (any) | 21.7% | -2.5% | 5 | 70.4 | 39.3 | 31.2 | 5 | 35.0% | 79.2% |
| Poisson noise floor | 13.4% | | | 44.2 | 21.9 | 22.3 | | | |

### Difference from the lowest-error run

Mean weekly difference from xgboost (secondary), with a 95% bootstrap interval (four-week blocks, 4,000 draws). An interval containing zero does not separate runs.

| Run | Net staffing error, point | Cost at r = 4, quantile |
|---|---:|---:|
| seasonal_naive (any) | +22.9 (+19.2 to +28.1) | +58.1 (+48.6 to +71.2) |
| mstl (primary) | +8.8 (+5.4 to +14.5) | +22.3 (+12.6 to +35.1) |
| mstl (secondary) | +9.1 (+5.6 to +14.7) | +22.8 (+13.2 to +35.8) |
| prophet (primary) | +14.8 (+8.5 to +24.2) | +36.0 (+18.5 to +62.0) |
| prophet (secondary) | +7.6 (+4.4 to +12.7) | +12.3 (+4.8 to +22.0) |
| xgboost (primary) | +1.9 (-0.0 to +4.1) | +5.4 (+1.3 to +9.9) |
| chronos2 (any) | +7.5 (+3.6 to +14.0) | +15.3 (+6.3 to +28.0) |
| chronos2_calendar (any) | +7.8 (+4.2 to +14.0) | +17.5 (+9.0 to +29.2) |

### By split

Net staffing error per week, and the share of understaffed intervals in brackets.

| Run | Peak | Off-peak | Post-closure |
|---|---:|---:|---:|
| seasonal_naive (any) | 36.0 (37%) | 49.5 (34%) | 3.7 (45%) |
| mstl (primary) | 30.7 (37%) | 40.7 (30%) | 3.1 (45%) |
| mstl (secondary) | 30.7 (37%) | 40.9 (30%) | 3.1 (46%) |
| prophet (primary) | 32.8 (36%) | 44.7 (31%) | 3.0 (49%) |
| prophet (secondary) | 28.9 (34%) | 41.3 (26%) | 2.8 (47%) |
| xgboost (primary) | 26.5 (31%) | 38.0 (26%) | 2.7 (37%) |
| xgboost (secondary) | 26.5 (35%) | 36.1 (26%) | 2.4 (42%) |
| chronos2 (any) | 30.2 (42%) | 40.0 (34%) | 3.4 (52%) |
| chronos2_calendar (any) | 30.3 (38%) | 40.1 (33%) | 3.4 (48%) |

## Cost-ratio sweep

Each run is staffed three ways at each ratio. Point uses the forecast as given. Uplift multiplies it by the q-quantile of actual / forecast ratios, pooled over open half-hours in 26 weekly calibration forecasts. Quantile uses the same method by slot type: weekday or weekend, peak or off-peak. The target is q = r / (1 + r). Calibration runs from 2025-02-24 to 2025-08-18, inside the training period. We score all methods on the separate test period.

Lowest weekly cost at each ratio, in units of one overstaffed agent-hour.

| r | q | Uplift (xgboost secondary) | Point | Uplift | Quantile |
|---:|---:|---:|---|---|---|
| 0.5 | 0.33 | 0.89 | 46.8 xgboost (secondary) | 42.0 xgboost (secondary) | 41.7 xgboost (secondary) |
| 1 | 0.50 | 0.99 | 62.6 xgboost (secondary) | 62.5 xgboost (secondary) | 62.1 xgboost (secondary) |
| 2 | 0.67 | 1.08 | 94.1 xgboost (primary) | 89.0 xgboost (secondary) | 88.8 xgboost (secondary) |
| 4 | 0.80 | 1.18 | 153.4 xgboost (primary) | 119.0 xgboost (secondary) | 118.3 xgboost (secondary) |
| 8 | 0.89 | 1.29 | 272.0 xgboost (primary) | 153.0 xgboost (secondary) | 152.6 xgboost (secondary) |
| 16 | 0.94 | 1.39 | 509.3 xgboost (primary) | 192.6 xgboost (secondary) | 191.7 xgboost (secondary) |

Quantile staffing cost relative to the lowest-cost run at that ratio (1.00 = lowest).

| Run | r = 0.5 | r = 1 | r = 2 | r = 4 | r = 8 | r = 16 |
|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive (any) | 1.33 | 1.38 | 1.43 | 1.49 | 1.56 | 1.61 |
| mstl (primary) | 1.15 | 1.15 | 1.15 | 1.19 | 1.19 | 1.18 |
| mstl (secondary) | 1.15 | 1.15 | 1.15 | 1.19 | 1.19 | 1.18 |
| prophet (primary) | 1.18 | 1.22 | 1.25 | 1.30 | 1.37 | 1.41 |
| prophet (secondary) | 1.09 | 1.09 | 1.10 | 1.10 | 1.11 | 1.09 |
| xgboost (primary) | 1.03 | 1.03 | 1.03 | 1.05 | 1.03 | 1.03 |
| xgboost (secondary) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| chronos2 (any) | 1.12 | 1.12 | 1.12 | 1.13 | 1.12 | 1.11 |
| chronos2_calendar (any) | 1.14 | 1.13 | 1.14 | 1.15 | 1.15 | 1.14 |

### Does accuracy predict cost?

Spearman rank correlation between WAPE and cost for all 9 runs and the 6 primary-training runs.

| r | Point, 9 runs | Quantile, 9 runs | Point, 6 primary | Quantile, 6 primary |
|---:|---:|---:|---:|---:|
| 0.5 | 0.80 | 1.00 | 0.94 | 1.00 |
| 1 | 0.98 | 0.98 | 1.00 | 1.00 |
| 2 | 0.83 | 1.00 | 0.77 | 1.00 |
| 4 | 0.83 | 1.00 | 0.77 | 1.00 |
| 8 | 0.77 | 1.00 | 0.60 | 1.00 |
| 16 | 0.77 | 1.00 | 0.60 | 1.00 |

### Calibration of the quantiles on the test window

Share of open intervals where actual demand is at or below the quantile forecast. A calibrated quantile matches its level.

| Run | q = 0.33 | q = 0.50 | q = 0.67 | q = 0.80 | q = 0.89 | q = 0.94 |
|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive (any) | 0.35 | 0.53 | 0.67 | 0.81 | 0.89 | 0.93 |
| mstl (primary) | 0.37 | 0.52 | 0.67 | 0.80 | 0.88 | 0.93 |
| mstl (secondary) | 0.37 | 0.52 | 0.67 | 0.80 | 0.87 | 0.93 |
| prophet (primary) | 0.36 | 0.51 | 0.64 | 0.76 | 0.85 | 0.91 |
| prophet (secondary) | 0.40 | 0.56 | 0.70 | 0.82 | 0.90 | 0.94 |
| xgboost (primary) | 0.36 | 0.53 | 0.68 | 0.81 | 0.89 | 0.94 |
| xgboost (secondary) | 0.38 | 0.53 | 0.67 | 0.79 | 0.87 | 0.93 |
| chronos2 (any) | 0.34 | 0.49 | 0.64 | 0.76 | 0.86 | 0.92 |
| chronos2_calendar (any) | 0.36 | 0.50 | 0.64 | 0.75 | 0.85 | 0.91 |

By quarter, q = 0.80. 2025Q3 is September only and 2026Q3 is July and August.

| Run | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 | 2026Q3 |
|---|---:|---:|---:|---:|---:|
| seasonal_naive (any) | 0.82 | 0.84 | 0.72 | 0.82 | 0.84 |
| mstl (primary) | 0.82 | 0.83 | 0.71 | 0.82 | 0.84 |
| mstl (secondary) | 0.82 | 0.82 | 0.71 | 0.82 | 0.84 |
| prophet (primary) | 0.85 | 0.85 | 0.51 | 0.87 | 0.79 |
| prophet (secondary) | 0.87 | 0.81 | 0.81 | 0.81 | 0.82 |
| xgboost (primary) | 0.84 | 0.78 | 0.75 | 0.82 | 0.91 |
| xgboost (secondary) | 0.92 | 0.75 | 0.71 | 0.79 | 0.89 |
| chronos2 (any) | 0.82 | 0.80 | 0.65 | 0.81 | 0.78 |
| chronos2_calendar (any) | 0.86 | 0.80 | 0.64 | 0.77 | 0.77 |

## Scenarios

Each scenario changes one input. Rank correlation compares WAPE with point-staffing error.

| Scenario | Best by WAPE | Best by agent-hours error | Rank correlation | Best, quantile, r = 4 |
|---|---|---|---:|---|
| primary | xgboost (secondary) | xgboost (secondary) | 0.98 | xgboost (secondary) |
| target 75% | xgboost (secondary) | xgboost (secondary) | 1.00 | xgboost (secondary) |
| patience 60 s | xgboost (secondary) | xgboost (secondary) | 0.97 | xgboost (secondary) |
| patience 300 s | xgboost (secondary) | xgboost (secondary) | 0.98 | xgboost (secondary) |
| service 360 s | xgboost (secondary) | xgboost (secondary) | 1.00 | xgboost (secondary) |
| Erlang C | xgboost (secondary) | xgboost (secondary) | 1.00 | xgboost (secondary) |
