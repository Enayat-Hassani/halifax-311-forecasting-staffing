# Baselines

52 weekly origins forecast the week starting seven days later, from 2025-09-01 to 2026-08-30. Open intervals only. Bias = (forecast − actual) / actual. MASE uses the primary-training Seasonal Naive MAE (8.42 calls). History weeks show how much history each model reads at the final origin.

This report compares forecast accuracy across nine Halifax 311 runs. The decision analysis tests how those rankings relate to staffing error and cost when understaffing costs more.

## All (7,696 intervals)

| Model | Window | History weeks | WAPE | MAE | RMSE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive | any | 1 | 26.1% | 8.30 | 11.25 | 0.985 | +0.5% |
| mstl | primary | 108 | 21.8% | 6.94 | 9.37 | 0.824 | -0.0% |
| mstl | secondary | 476 | 21.8% | 6.95 | 9.38 | 0.825 | -0.1% |
| prophet | primary | 108 | 23.5% | 7.49 | 10.16 | 0.889 | -1.0% |
| prophet | secondary | 476 | 21.4% | 6.79 | 9.07 | 0.807 | +0.8% |
| xgboost | primary | 108 | 19.8% | 6.30 | 8.51 | 0.748 | +1.5% |
| xgboost | secondary | 476 | 19.3% | 6.13 | 8.29 | 0.728 | -0.0% |
| chronos2 | any | 24 | 21.5% | 6.84 | 9.34 | 0.812 | -4.4% |
| chronos2_calendar | any | 24 | 21.7% | 6.90 | 9.39 | 0.819 | -2.5% |

## Peak (2,717 intervals)

| Model | Window | History weeks | WAPE | MAE | RMSE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive | any | 1 | 21.9% | 10.41 | 13.35 | 1.236 | +0.6% |
| mstl | primary | 108 | 18.6% | 8.85 | 11.18 | 1.051 | +0.1% |
| mstl | secondary | 476 | 18.6% | 8.85 | 11.18 | 1.051 | +0.0% |
| prophet | primary | 108 | 20.0% | 9.47 | 11.89 | 1.125 | +0.2% |
| prophet | secondary | 476 | 17.6% | 8.36 | 10.59 | 0.993 | +0.1% |
| xgboost | primary | 108 | 16.3% | 7.73 | 9.76 | 0.918 | +1.3% |
| xgboost | secondary | 476 | 16.1% | 7.64 | 9.65 | 0.907 | -0.7% |
| chronos2 | any | 24 | 18.4% | 8.74 | 11.18 | 1.038 | -3.5% |
| chronos2_calendar | any | 24 | 18.6% | 8.81 | 11.21 | 1.046 | -1.2% |

## Off peak (4,979 intervals)

| Model | Window | History weeks | WAPE | MAE | RMSE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive | any | 1 | 30.7% | 7.14 | 9.92 | 0.848 | +0.5% |
| mstl | primary | 108 | 25.3% | 5.89 | 8.22 | 0.700 | -0.1% |
| mstl | secondary | 476 | 25.4% | 5.91 | 8.23 | 0.702 | -0.2% |
| prophet | primary | 108 | 27.5% | 6.40 | 9.07 | 0.760 | -2.3% |
| prophet | secondary | 476 | 25.5% | 5.94 | 8.12 | 0.705 | +1.6% |
| xgboost | primary | 108 | 23.7% | 5.52 | 7.75 | 0.655 | +1.8% |
| xgboost | secondary | 476 | 22.8% | 5.30 | 7.45 | 0.630 | +0.7% |
| chronos2 | any | 24 | 24.9% | 5.79 | 8.17 | 0.688 | -5.3% |
| chronos2_calendar | any | 24 | 25.1% | 5.85 | 8.23 | 0.695 | -3.9% |

## Post closure (264 intervals)

| Model | Window | History weeks | WAPE | MAE | RMSE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive | any | 1 | 24.6% | 10.15 | 12.85 | 1.206 | -5.3% |
| mstl | primary | 108 | 21.1% | 8.71 | 11.01 | 1.035 | -5.9% |
| mstl | secondary | 476 | 21.2% | 8.74 | 11.04 | 1.038 | -5.9% |
| prophet | primary | 108 | 20.3% | 8.35 | 10.40 | 0.991 | -8.0% |
| prophet | secondary | 476 | 20.1% | 8.26 | 10.49 | 0.981 | -8.9% |
| xgboost | primary | 108 | 18.6% | 7.65 | 9.52 | 0.908 | -0.3% |
| xgboost | secondary | 476 | 16.8% | 6.93 | 8.66 | 0.823 | -5.8% |
| chronos2 | any | 24 | 24.1% | 9.91 | 12.51 | 1.177 | -12.2% |
| chronos2_calendar | any | 24 | 22.9% | 9.43 | 12.02 | 1.120 | -8.8% |

## By month: WAPE (bias)

Seasonal bias is hidden in the annual total.

| | seasonal_naive (any) | mstl (primary) | mstl (secondary) | prophet (primary) | prophet (secondary) | xgboost (primary) | xgboost (secondary) | chronos2 (any) | chronos2_calendar (any) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-09 | 21.4% (-3.8%) | 16.9% (-3.1%) | 16.8% (-3.4%) | 17.5% (+2.5%) | 17.3% (+1.3%) | 17.0% (+2.9%) | 18.4% (+10.6%) | 17.2% (-5.8%) | 16.4% (+0.3%) |
| 2025-10 | 21.7% (-6.0%) | 19.5% (-8.8%) | 19.5% (-8.7%) | 19.8% (-11.9%) | 23.6% (-12.8%) | 18.0% (+1.0%) | 16.8% (-6.1%) | 18.4% (-10.0%) | 18.0% (-7.0%) |
| 2025-11 | 32.5% (+21.5%) | 29.1% (+22.4%) | 29.0% (+22.0%) | 35.8% (+32.2%) | 22.6% (+10.5%) | 19.9% (-0.4%) | 18.3% (+1.0%) | 31.1% (+23.4%) | 31.5% (+24.4%) |
| 2025-12 | 31.1% (+8.9%) | 30.8% (+11.8%) | 30.7% (+11.2%) | 32.4% (+21.7%) | 31.1% (+18.7%) | 25.7% (+3.8%) | 24.0% (-2.2%) | 24.8% (-3.3%) | 24.6% (-2.3%) |
| 2026-01 | 31.5% (-17.6%) | 25.1% (-16.6%) | 25.1% (-16.7%) | 33.5% (-31.1%) | 22.6% (+0.9%) | 22.7% (-11.1%) | 22.2% (-9.0%) | 26.3% (-21.3%) | 26.5% (-20.7%) |
| 2026-02 | 28.8% (+4.7%) | 24.7% (+1.3%) | 24.8% (+1.8%) | 31.6% (-25.6%) | 26.9% (+0.8%) | 27.1% (+4.1%) | 25.3% (-6.7%) | 23.6% (-3.5%) | 24.1% (-2.7%) |
| 2026-03 | 30.5% (-1.9%) | 24.3% (-2.5%) | 24.4% (-2.5%) | 20.6% (-5.0%) | 21.4% (+2.7%) | 19.6% (-1.2%) | 19.4% (-0.7%) | 24.2% (-11.1%) | 24.2% (-7.3%) |
| 2026-04 | 23.8% (-0.9%) | 19.8% (-3.1%) | 19.9% (-2.9%) | 18.3% (-1.8%) | 20.1% (-8.8%) | 17.8% (-5.1%) | 18.4% (-9.4%) | 19.8% (-6.8%) | 20.3% (-6.7%) |
| 2026-05 | 25.1% (+5.7%) | 21.1% (+4.9%) | 21.1% (+4.7%) | 24.2% (+16.2%) | 19.3% (+2.8%) | 19.8% (+6.9%) | 18.4% (+3.8%) | 19.1% (+4.5%) | 20.2% (+5.6%) |
| 2026-06 | 21.8% (-1.5%) | 15.9% (-0.5%) | 15.9% (-0.7%) | 16.6% (+3.0%) | 16.0% (+0.5%) | 14.4% (+1.1%) | 15.0% (+4.6%) | 14.9% (-5.3%) | 15.3% (-5.5%) |
| 2026-07 | 23.5% (+3.0%) | 19.0% (+1.2%) | 19.2% (+1.4%) | 18.4% (-3.4%) | 17.5% (-1.8%) | 17.7% (+5.8%) | 17.5% (+5.7%) | 19.9% (-1.3%) | 21.5% (+0.8%) |
| 2026-08 | 24.6% (+1.6%) | 19.1% (+1.0%) | 19.1% (+0.9%) | 19.0% (+2.3%) | 19.7% (+3.8%) | 20.5% (+12.3%) | 19.6% (+10.1%) | 21.5% (-5.7%) | 20.7% (-3.4%) |

## By day type: WAPE (bias)

A surge or lull day is in the top or bottom tenth of the test year relative to its weekday median. Sorting on actuals pushes surge-day bias down for any forecast.

| | seasonal_naive (any) | mstl (primary) | mstl (secondary) | prophet (primary) | prophet (secondary) | xgboost (primary) | xgboost (secondary) | chronos2 (any) | chronos2_calendar (any) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lull | 37.7% (+20.6%) | 34.5% (+24.7%) | 34.5% (+24.5%) | 34.1% (+19.1%) | 39.6% (+34.7%) | 31.6% (+23.9%) | 28.4% (+18.0%) | 26.2% (+12.8%) | 27.7% (+14.9%) |
| normal | 25.1% (+1.4%) | 20.3% (+0.8%) | 20.3% (+0.7%) | 21.9% (+0.9%) | 19.2% (+1.2%) | 18.6% (+1.8%) | 18.1% (+1.2%) | 20.7% (-3.4%) | 20.9% (-1.5%) |
| surge | 26.4% (-16.6%) | 24.7% (-19.7%) | 24.7% (-19.6%) | 28.8% (-25.5%) | 25.5% (-21.2%) | 21.4% (-13.3%) | 22.1% (-18.5%) | 24.4% (-20.7%) | 23.9% (-19.3%) |

## Run cost

Wall time of one forecast call on an Apple M2 CPU, comparable between models only. The first call of Chronos-2 includes loading the model.

| Model | Window | First (s) | Median (s) | 52 origins (min) |
|---|---|---:|---:|---:|
| seasonal_naive | any | 0.0 | 0.0 | 0.0 |
| mstl | primary | 22.0 | 14.8 | 13.0 |
| mstl | secondary | 46.0 | 46.4 | 40.1 |
| prophet | primary | 23.0 | 16.5 | 13.9 |
| prophet | secondary | 58.7 | 60.4 | 55.4 |
| xgboost | primary | 1.1 | 1.0 | 0.9 |
| xgboost | secondary | 1.5 | 1.4 | 1.3 |
| chronos2 | any | 13.9 | 0.3 | 0.5 |
| chronos2_calendar | any | 8.7 | 0.7 | 0.7 |
