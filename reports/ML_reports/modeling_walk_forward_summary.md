# Walk-Forward Validation Summary

## Purpose

This script evaluates baseline vs baseline+trends using expanding-window walk-forward validation on the same aligned sample.
It tests whether lagged Google Trends improves prediction consistently across multiple future periods, not just one holdout split.

## Inputs used

- Master table: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\data\master\hotel_master_table.xlsx`
- Best lag file: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\eda_outputs\best_lag_correlations.csv`
- Included lagged Trends: trends_turkiye_side_otel_lag_28, trends_turkiye_side_otel_lag_21, trends_turkiye_side_otel_lag_14, trends_germany_alanya_hotel_lag_7, trends_germany_alanya_hotel_lag_28, trends_turkiye_side_otel_lag_7, trends_germany_alanya_hotel_lag_21, trends_turkiye_alanya_otel_fiyatlari_lag_21

## Fold design

- Fold 1: train 2023-05-19 to 2024-08-19, test 2024-08-20 to 2024-10-27
- Fold 2: train 2023-05-19 to 2024-10-27, test 2024-10-28 to 2025-06-25
- Fold 3: train 2023-05-19 to 2025-06-25, test 2025-06-26 to 2025-09-01
- Fold 4: train 2023-05-19 to 2025-09-01, test 2025-09-02 to 2025-11-08

## Fold-level results

|   fold | dataset                                | model        |   train_rows |   test_rows | test_start_date   | test_end_date   |      MAE |     RMSE |        R2 |
|-------:|:---------------------------------------|:-------------|-------------:|------------:|:------------------|:----------------|---------:|---------:|----------:|
|      1 | baseline_calendar_autoreg_walk_forward | RandomForest |          538 |         138 | 2024-08-20        | 2024-10-27      |  3.35528 |  5.38565 | -2.56952  |
|      1 | baseline_calendar_autoreg_walk_forward | Ridge        |          538 |         138 | 2024-08-20        | 2024-10-27      |  6.70266 |  8.70096 | -8.31683  |
|      1 | baseline_plus_trends_walk_forward      | RandomForest |          538 |         138 | 2024-08-20        | 2024-10-27      |  3.26006 |  5.22487 | -2.35958  |
|      1 | baseline_plus_trends_walk_forward      | Ridge        |          538 |         138 | 2024-08-20        | 2024-10-27      |  4.69922 |  6.49897 | -4.19784  |
|      2 | baseline_calendar_autoreg_walk_forward | RandomForest |          676 |         129 | 2024-10-28        | 2025-06-25      | 11.6386  | 16.1954  |  0.464053 |
|      2 | baseline_calendar_autoreg_walk_forward | Ridge        |          676 |         129 | 2024-10-28        | 2025-06-25      | 14.6911  | 21.5429  |  0.051696 |
|      2 | baseline_plus_trends_walk_forward      | RandomForest |          676 |         129 | 2024-10-28        | 2025-06-25      | 11.6909  | 16.1299  |  0.46838  |
|      2 | baseline_plus_trends_walk_forward      | Ridge        |          676 |         129 | 2024-10-28        | 2025-06-25      | 14.0495  | 19.8254  |  0.196878 |
|      3 | baseline_calendar_autoreg_walk_forward | RandomForest |          805 |         136 | 2025-06-26        | 2025-09-01      |  3.31444 |  5.02974 |  0.213081 |
|      3 | baseline_calendar_autoreg_walk_forward | Ridge        |          805 |         136 | 2025-06-26        | 2025-09-01      |  5.47563 |  6.9675  | -0.510057 |
|      3 | baseline_plus_trends_walk_forward      | RandomForest |          805 |         136 | 2025-06-26        | 2025-09-01      |  3.30845 |  4.98576 |  0.226782 |
|      3 | baseline_plus_trends_walk_forward      | Ridge        |          805 |         136 | 2025-06-26        | 2025-09-01      |  4.94237 |  6.45671 | -0.296766 |
|      4 | baseline_calendar_autoreg_walk_forward | RandomForest |          941 |         136 | 2025-09-02        | 2025-11-08      |  3.89217 |  6.05138 |  0.942797 |
|      4 | baseline_calendar_autoreg_walk_forward | Ridge        |          941 |         136 | 2025-09-02        | 2025-11-08      | 11.6427  | 18.0383  |  0.491724 |
|      4 | baseline_plus_trends_walk_forward      | RandomForest |          941 |         136 | 2025-09-02        | 2025-11-08      |  3.8773  |  5.81847 |  0.947116 |
|      4 | baseline_plus_trends_walk_forward      | Ridge        |          941 |         136 | 2025-09-02        | 2025-11-08      |  9.38452 | 16.2242  |  0.588818 |

## Mean results across folds

| dataset                                | model        |   folds |   mean_MAE |   std_MAE |   mean_RMSE |   std_RMSE |   mean_R2 |   std_R2 |
|:---------------------------------------|:-------------|--------:|-----------:|----------:|------------:|-----------:|----------:|---------:|
| baseline_calendar_autoreg_walk_forward | RandomForest |       4 |    5.55013 |   4.06753 |     8.16554 |    5.36996 | -0.237398 |  1.58394 |
| baseline_calendar_autoreg_walk_forward | Ridge        |       4 |    9.62802 |   4.30092 |    13.8124  |    7.08516 | -2.07087  |  4.18411 |
| baseline_plus_trends_walk_forward      | RandomForest |       4 |    5.53418 |   4.11404 |     8.03975 |    5.40478 | -0.179326 |  1.48401 |
| baseline_plus_trends_walk_forward      | Ridge        |       4 |    8.2689  |   4.41467 |    12.2513  |    6.82683 | -0.927229 |  2.21031 |

## By-hotel results across folds

|   fold | dataset                                | model        | hotel_name      |   test_rows |      MAE |     RMSE |          R2 |
|-------:|:---------------------------------------|:-------------|:----------------|------------:|---------:|---------:|------------:|
|      1 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          69 |  2.38329 |  3.29114 | -0.306129   |
|      1 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          69 |  4.32726 |  6.86869 | -4.96318    |
|      1 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          69 |  6.53328 |  8.508   | -7.72867    |
|      1 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          69 |  6.87204 |  8.88972 | -8.98867    |
|      1 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          69 |  2.37496 |  3.37029 | -0.36971    |
|      1 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          69 |  4.14517 |  6.57569 | -4.46529    |
|      1 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          69 |  4.59966 |  6.10632 | -3.49627    |
|      1 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          69 |  4.79879 |  6.86921 | -4.9641     |
|      2 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          61 | 10.2708  | 12.7413  |  0.227988   |
|      2 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          68 | 12.8657  | 18.7604  |  0.480755   |
|      2 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          61 | 11.3987  | 14.7875  | -0.0398985  |
|      2 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          68 | 17.6445  | 26.1584  | -0.00950255 |
|      2 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          61 | 10.1604  | 12.851   |  0.214632   |
|      2 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          68 | 13.0638  | 18.5854  |  0.4904     |
|      2 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          61 | 10.5741  | 13.3685  |  0.150111   |
|      2 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          68 | 17.1671  | 24.1932  |  0.136476   |
|      3 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          68 |  2.56665 |  3.55295 |  0.270441   |
|      3 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          68 |  4.06222 |  6.16223 |  0.187037   |
|      3 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          68 |  5.64935 |  6.52634 | -1.46162    |
|      3 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          68 |  5.30192 |  7.38235 | -0.166767   |
|      3 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          68 |  2.74054 |  3.70553 |  0.206435   |
|      3 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          68 |  3.87637 |  5.99872 |  0.229606   |
|      3 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          68 |  4.92983 |  6.09631 | -1.14791    |
|      3 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          68 |  4.9549  |  6.79803 |  0.0106233  |
|      4 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          68 |  2.8924  |  3.81661 |  0.975671   |
|      4 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          68 |  4.89193 |  7.65976 |  0.913481   |
|      4 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          68 | 11.9136  | 18.0199  |  0.521166   |
|      4 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          68 | 11.3718  | 18.0566  |  0.455435   |
|      4 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          68 |  2.94223 |  3.75315 |  0.976473   |
|      4 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          68 |  4.81237 |  7.32278 |  0.920926   |
|      4 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          68 |  9.15503 | 16.2188  |  0.560648   |
|      4 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          68 |  9.61401 | 16.2295  |  0.611588   |

## Interpretation

- Best baseline-only mean RMSE across folds: **8.166**.
- Best baseline+trends mean RMSE across folds: **8.040**.
- Mean RMSE difference (baseline - trends): **0.126**. Positive means the trends model performed better on average across folds.
- This result is stronger than a single holdout because it checks whether the same conclusion survives across multiple future windows.