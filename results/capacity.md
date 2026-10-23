# Capacity horizon

This report extends the staffing comparison to weekly forecasts 13 and 26 weeks ahead.

The test covers 52 weeks, 2025-09-01 to 2026-08-30. Forecasts use the secondary training window and open intervals. Each forecast starts Monday, `lead` weeks before its target. Bias = (forecast − actual) / actual. Prophet, XGBoost and calendar Chronos-2 use the known closure count for each target week. Other runs do not.

## Weekly accuracy at the scheduling lead

One-week-ahead forecasts, summed over open intervals.

| Run | WAPE | Bias |
|---|---:|---:|
| seasonal_naive_primary | 11.9% | +0.5% |
| mstl_primary | 11.1% | -0.0% |
| mstl_secondary | 11.2% | -0.1% |
| prophet_primary | 14.3% | -1.0% |
| prophet_secondary | 8.5% | +0.8% |
| xgboost_primary | 8.7% | +1.5% |
| xgboost_secondary | 8.7% | -0.0% |
| chronos2_primary | 12.2% | -4.4% |
| chronos2_calendar_primary | 12.6% | -2.5% |

## Weekly accuracy at the capacity leads

| Model | WAPE 13 w | Bias 13 w | WAPE 26 w | Bias 26 w |
|---|---:|---:|---:|---:|
| seasonal_naive | 11.7% | +1.1% | 11.7% | +1.1% |
| prophet | 10.1% | +2.1% | 9.8% | +1.9% |
| xgboost | 10.1% | -0.6% | 9.3% | -2.8% |
| chronos2 | 11.6% | +1.0% | 11.4% | +0.8% |
| chronos2_calendar | 10.7% | +0.9% | 11.0% | +0.5% |

## FTE

FTE = volume × handle time ÷ 3,600 ÷ occupancy ÷ (1 − shrinkage) ÷ paid hours. Assumptions: 237 s handle time, 85% occupancy, 32.5% shrinkage, and 37.5 paid hours per week. Weekly volume averages 14.4 FTE (5.6–19.9), or 21.5 at 353 s handle time. The estimate excludes interval-level safety staffing and shift inefficiency. It is not a final headcount plan. The assumptions scale estimates equally and do not change rankings.

### Weekly cost at 13 weeks, FTE

Mean weekly cost = r × short FTE + spare FTE. Uplift uses a pooled quantile of actual / forecast. Calibration uses 52 weeks at the same lead.

| Model | Method | r = 0.5 | r = 1 | r = 2 | r = 4 | r = 8 | r = 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive | point | 1.30 | 1.68 | 2.44 | 3.97 | 7.03 | 13.13 |
| seasonal_naive | uplift | 1.20 | 1.73 | 2.40 | 3.13 | 4.14 | 4.57 |
| prophet | point | 1.17 | 1.46 | 2.04 | 3.20 | 5.51 | 10.14 |
| prophet | uplift | 0.95 | 1.40 | 2.05 | 2.77 | 3.46 | 4.26 |
| xgboost | point | 1.07 | 1.45 | 2.22 | 3.77 | 6.86 | 13.04 |
| xgboost | uplift | 1.00 | 1.51 | 2.11 | 2.79 | 3.66 | 4.74 |
| chronos2 | point | 1.29 | 1.67 | 2.43 | 3.96 | 7.01 | 13.12 |
| chronos2 | uplift | 1.18 | 1.72 | 2.37 | 3.23 | 4.11 | 4.88 |
| chronos2_calendar | point | 1.18 | 1.54 | 2.24 | 3.66 | 6.49 | 12.14 |
| chronos2_calendar | uplift | 1.08 | 1.52 | 2.21 | 3.10 | 3.83 | 4.58 |

### Weekly cost at 26 weeks, FTE

Mean weekly cost = r × short FTE + spare FTE. Uplift uses a pooled quantile of actual / forecast. Calibration uses 52 weeks at the same lead.

| Model | Method | r = 0.5 | r = 1 | r = 2 | r = 4 | r = 8 | r = 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| seasonal_naive | point | 1.30 | 1.68 | 2.44 | 3.97 | 7.03 | 13.13 |
| seasonal_naive | uplift | 1.20 | 1.73 | 2.40 | 3.13 | 4.14 | 4.57 |
| prophet | point | 1.12 | 1.41 | 1.97 | 3.11 | 5.38 | 9.92 |
| prophet | uplift | 0.89 | 1.33 | 1.97 | 2.64 | 3.35 | 4.28 |
| xgboost | point | 0.90 | 1.34 | 2.22 | 3.96 | 7.46 | 14.45 |
| xgboost | uplift | 0.90 | 1.36 | 2.00 | 2.84 | 3.90 | 4.02 |
| chronos2 | point | 1.26 | 1.64 | 2.40 | 3.92 | 6.97 | 13.07 |
| chronos2 | uplift | 1.16 | 1.65 | 2.29 | 3.11 | 4.00 | 4.53 |
| chronos2_calendar | point | 1.21 | 1.59 | 2.34 | 3.86 | 6.88 | 12.94 |
| chronos2_calendar | uplift | 1.07 | 1.57 | 2.25 | 3.16 | 4.04 | 5.41 |
