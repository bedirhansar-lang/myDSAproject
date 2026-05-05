# Fair Same-Window Modeling Summary

## Purpose

This script compares baseline vs baseline+trends on the exact same rows and the exact same time split.
It addresses the fairness issue in the first-pass modeling comparison, where each setup dropped missing rows separately before splitting.

## Inputs used

- Master table: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\data\master\hotel_master_table.xlsx`
- Best lag file: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\eda_outputs\best_lag_correlations.csv`
- Included lagged Trends: trends_turkiye_side_otel_lag_28, trends_turkiye_side_otel_lag_21, trends_turkiye_side_otel_lag_14, trends_germany_alanya_hotel_lag_7, trends_germany_alanya_hotel_lag_28, trends_turkiye_side_otel_lag_7, trends_germany_alanya_hotel_lag_21, trends_turkiye_alanya_otel_fiyatlari_lag_21

## Shared split design

- Common train rows: **857**
- Common test rows: **220**
- Test period starts on: **2025-07-22**

## Model comparison

| dataset                               | model        |   train_rows |   test_rows | test_start_date   | test_end_date   |     MAE |     RMSE |       R2 |
|:--------------------------------------|:-------------|-------------:|------------:|:------------------|:----------------|--------:|---------:|---------:|
| baseline_calendar_autoreg_same_window | RandomForest |          857 |         220 | 2025-07-22        | 2025-11-08      | 3.15454 |  4.97429 | 0.941513 |
| baseline_calendar_autoreg_same_window | Ridge        |          857 |         220 | 2025-07-22        | 2025-11-08      | 8.86565 | 14.4775  | 0.50457  |
| baseline_plus_trends_same_window      | RandomForest |          857 |         220 | 2025-07-22        | 2025-11-08      | 3.1646  |  4.87746 | 0.943768 |
| baseline_plus_trends_same_window      | Ridge        |          857 |         220 | 2025-07-22        | 2025-11-08      | 7.42493 | 13.0384  | 0.598164 |

## By-hotel comparison

| dataset                               | model        | hotel_name      |   test_rows |     MAE |     RMSE |       R2 |
|:--------------------------------------|:-------------|:----------------|------------:|--------:|---------:|---------:|
| baseline_calendar_autoreg_same_window | RandomForest | Side Mare Hotel |         110 | 2.3653  |  3.11136 | 0.975273 |
| baseline_calendar_autoreg_same_window | RandomForest | Azura Deluxe    |         110 | 3.94378 |  6.30924 | 0.912105 |
| baseline_calendar_autoreg_same_window | Ridge        | Azura Deluxe    |         110 | 8.69458 | 14.3576  | 0.544827 |
| baseline_calendar_autoreg_same_window | Ridge        | Side Mare Hotel |         110 | 9.03673 | 14.5963  | 0.455797 |
| baseline_plus_trends_same_window      | RandomForest | Side Mare Hotel |         110 | 2.45395 |  3.15736 | 0.974536 |
| baseline_plus_trends_same_window      | RandomForest | Azura Deluxe    |         110 | 3.87526 |  6.13272 | 0.916954 |
| baseline_plus_trends_same_window      | Ridge        | Azura Deluxe    |         110 | 7.46514 | 12.9974  | 0.626988 |
| baseline_plus_trends_same_window      | Ridge        | Side Mare Hotel |         110 | 7.38471 | 13.0794  | 0.563033 |

## Interpretation

- Best baseline-only RMSE on the shared window: **4.974**.
- Best baseline+trends RMSE on the shared window: **4.877**.
- RMSE difference (baseline - trends): **0.097**. Positive means the trends model performed better on the same window.
- This result is methodologically cleaner than the first-pass comparison because both systems are evaluated on the same sample and same future period.