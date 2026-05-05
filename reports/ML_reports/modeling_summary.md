# Baseline Modeling Summary

## Setup

- Target: `occupancy_rate`
- Time-aware split: final 20% of unique dates used as test set.
- Baseline features: hotel identity, calendar seasonality, and occupancy lags (7/14/28 days).
- Trends model: baseline features plus top lagged Google Trends features selected from the first EDA pass.
- Top lagged Trends included: trends_turkiye_side_otel_lag_28, trends_turkiye_side_otel_lag_21, trends_turkiye_side_otel_lag_14, trends_germany_alanya_hotel_lag_7, trends_germany_alanya_hotel_lag_28, trends_turkiye_side_otel_lag_7, trends_germany_alanya_hotel_lag_21, trends_turkiye_alanya_otel_fiyatlari_lag_21

## Model comparison

| dataset                   | model        |   train_rows |   test_rows | test_start_date   | test_end_date   |     MAE |     RMSE |       R2 |
|:--------------------------|:-------------|-------------:|------------:|:------------------|:----------------|--------:|---------:|---------:|
| baseline_calendar_autoreg | RandomForest |          983 |         268 | 2025-06-28        | 2025-11-08      | 3.80946 |  5.87133 | 0.903819 |
| baseline_calendar_autoreg | Ridge        |          983 |         268 | 2025-06-28        | 2025-11-08      | 9.48504 | 14.5748  | 0.40732  |
| baseline_plus_trends      | RandomForest |          857 |         220 | 2025-07-22        | 2025-11-08      | 3.1646  |  4.87746 | 0.943768 |
| baseline_plus_trends      | Ridge        |          857 |         220 | 2025-07-22        | 2025-11-08      | 7.42493 | 13.0384  | 0.598164 |

## By-hotel performance

| dataset                   | model        | hotel_name      |   test_rows |     MAE |     RMSE |       R2 |
|:--------------------------|:-------------|:----------------|------------:|--------:|---------:|---------:|
| baseline_calendar_autoreg | RandomForest | Side Mare Hotel |         134 | 2.86767 |  3.90744 | 0.953194 |
| baseline_calendar_autoreg | RandomForest | Azura Deluxe    |         134 | 4.75124 |  7.32645 | 0.862011 |
| baseline_calendar_autoreg | Ridge        | Azura Deluxe    |         134 | 9.09684 | 14.4535  | 0.462965 |
| baseline_calendar_autoreg | Ridge        | Side Mare Hotel |         134 | 9.87325 | 14.695   | 0.338    |
| baseline_plus_trends      | RandomForest | Side Mare Hotel |         110 | 2.45395 |  3.15736 | 0.974536 |
| baseline_plus_trends      | RandomForest | Azura Deluxe    |         110 | 3.87526 |  6.13272 | 0.916954 |
| baseline_plus_trends      | Ridge        | Azura Deluxe    |         110 | 7.46514 | 12.9974  | 0.626988 |
| baseline_plus_trends      | Ridge        | Side Mare Hotel |         110 | 7.38471 | 13.0794  | 0.563033 |

## Interpretation

- Best overall test RMSE: **4.877** from **baseline_plus_trends / RandomForest**.
- Best baseline-only RMSE: **5.871**.
- Best trends-augmented RMSE: **4.877**.
- RMSE difference (baseline - trends): **0.994**. Positive means Trends helped.
- This is still a first-pass result. It tests whether lagged Google Trends adds incremental signal beyond hotel seasonality and recent occupancy.