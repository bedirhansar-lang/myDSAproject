# Naive Benchmark Comparison Summary

## Purpose

This script compares the learned baseline and baseline+trends models against two simple rule-based time-series benchmarks: naive persistence and seasonal naive (7-day).

## Inputs used

- Master table: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\data\master\hotel_master_table.xlsx`
- Best lag file: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\eda_outputs\best_lag_correlations.csv`
- Included lagged Trends: trends_turkiye_side_otel_lag_28, trends_turkiye_side_otel_lag_21, trends_turkiye_side_otel_lag_14, trends_germany_alanya_hotel_lag_7, trends_germany_alanya_hotel_lag_28, trends_turkiye_side_otel_lag_7, trends_germany_alanya_hotel_lag_21, trends_turkiye_alanya_otel_fiyatlari_lag_21

## Fair same-window comparison

- Shared test period starts on: **2025-07-22**
| dataset                               | model        |     MAE |     RMSE |       R2 |   train_rows |   test_rows | test_start_date   | test_end_date   |
|:--------------------------------------|:-------------|--------:|---------:|---------:|-------------:|------------:|:------------------|:----------------|
| NaivePersistence                      | RuleBased    | 2.72107 |  4.06805 | 0.960883 |          857 |         220 | 2025-07-22        | 2025-11-08      |
| baseline_plus_trends_same_window      | RandomForest | 3.1646  |  4.87746 | 0.943768 |          857 |         220 | 2025-07-22        | 2025-11-08      |
| baseline_calendar_autoreg_same_window | RandomForest | 3.15454 |  4.97429 | 0.941513 |          857 |         220 | 2025-07-22        | 2025-11-08      |
| baseline_plus_trends_same_window      | Ridge        | 7.42493 | 13.0384  | 0.598164 |          857 |         220 | 2025-07-22        | 2025-11-08      |
| baseline_calendar_autoreg_same_window | Ridge        | 8.86565 | 14.4775  | 0.50457  |          857 |         220 | 2025-07-22        | 2025-11-08      |
| SeasonalNaive7                        | RuleBased    | 7.39885 | 16.7107  | 0.339935 |          857 |         220 | 2025-07-22        | 2025-11-08      |

### By-hotel same-window comparison

| dataset                               | model        | hotel_name      |   test_rows |     MAE |     RMSE |       R2 |
|:--------------------------------------|:-------------|:----------------|------------:|--------:|---------:|---------:|
| baseline_calendar_autoreg_same_window | RandomForest | Side Mare Hotel |         110 | 2.3653  |  3.11136 | 0.975273 |
| baseline_plus_trends_same_window      | RandomForest | Side Mare Hotel |         110 | 2.45395 |  3.15736 | 0.974536 |
| NaivePersistence                      | RuleBased    | Side Mare Hotel |         110 | 2.54259 |  3.99165 | 0.959301 |
| NaivePersistence                      | RuleBased    | Azura Deluxe    |         110 | 2.89955 |  4.14304 | 0.962099 |
| baseline_plus_trends_same_window      | RandomForest | Azura Deluxe    |         110 | 3.87526 |  6.13272 | 0.916954 |
| baseline_calendar_autoreg_same_window | RandomForest | Azura Deluxe    |         110 | 3.94378 |  6.30924 | 0.912105 |
| baseline_plus_trends_same_window      | Ridge        | Azura Deluxe    |         110 | 7.46514 | 12.9974  | 0.626988 |
| baseline_plus_trends_same_window      | Ridge        | Side Mare Hotel |         110 | 7.38471 | 13.0794  | 0.563033 |
| baseline_calendar_autoreg_same_window | Ridge        | Azura Deluxe    |         110 | 8.69458 | 14.3576  | 0.544827 |
| baseline_calendar_autoreg_same_window | Ridge        | Side Mare Hotel |         110 | 9.03673 | 14.5963  | 0.455797 |
| SeasonalNaive7                        | RuleBased    | Azura Deluxe    |         110 | 7.78345 | 16.1222  | 0.426074 |
| SeasonalNaive7                        | RuleBased    | Side Mare Hotel |         110 | 7.01425 | 17.2792  | 0.237353 |

## Walk-forward comparison

| dataset                                | model        |   folds |   mean_MAE |   std_MAE |   mean_RMSE |   std_RMSE |   mean_R2 |   std_R2 |
|:---------------------------------------|:-------------|--------:|-----------:|----------:|------------:|-----------:|----------:|---------:|
| NaivePersistence                       | RuleBased    |       4 |    2.99715 |   1.10907 |     4.17007 |    1.50248 |  0.647425 | 0.44999  |
| baseline_plus_trends_walk_forward      | RandomForest |       4 |    5.53418 |   4.11404 |     8.03975 |    5.40478 | -0.179326 | 1.48401  |
| baseline_calendar_autoreg_walk_forward | RandomForest |       4 |    5.55013 |   4.06753 |     8.16554 |    5.36996 | -0.237398 | 1.58394  |
| baseline_plus_trends_walk_forward      | Ridge        |       4 |    8.2689  |   4.41467 |    12.2513  |    6.82683 | -0.927229 | 2.21031  |
| baseline_calendar_autoreg_walk_forward | Ridge        |       4 |    9.62802 |   4.30092 |    13.8124  |    7.08516 | -2.07087  | 4.18411  |
| SeasonalNaive7                         | RuleBased    |       4 |    9.08156 |   6.90803 |    14.5373  |   10.4699  | -0.391287 | 0.521823 |

### By-hotel walk-forward comparison

|   fold | dataset                                | model        | hotel_name      |   test_rows |      MAE |     RMSE |          R2 |
|-------:|:---------------------------------------|:-------------|:----------------|------------:|---------:|---------:|------------:|
|      1 | NaivePersistence                       | RuleBased    | Side Mare Hotel |          69 |  2.07432 |  2.83679 |  0.0296091  |
|      1 | NaivePersistence                       | RuleBased    | Azura Deluxe    |          69 |  2.17623 |  2.879   | -0.0476481  |
|      1 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          69 |  2.38329 |  3.29114 | -0.306129   |
|      1 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          69 |  2.37496 |  3.37029 | -0.36971    |
|      1 | SeasonalNaive7                         | RuleBased    | Azura Deluxe    |          69 |  2.85087 |  3.58836 | -0.627511   |
|      1 | SeasonalNaive7                         | RuleBased    | Side Mare Hotel |          69 |  2.80797 |  3.65249 | -0.608687   |
|      1 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          69 |  4.59966 |  6.10632 | -3.49627    |
|      1 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          69 |  4.14517 |  6.57569 | -4.46529    |
|      1 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          69 |  4.32726 |  6.86869 | -4.96318    |
|      1 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          69 |  4.79879 |  6.86921 | -4.9641     |
|      1 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          69 |  6.53328 |  8.508   | -7.72867    |
|      1 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          69 |  6.87204 |  8.88972 | -8.98867    |
|      2 | NaivePersistence                       | RuleBased    | Side Mare Hotel |          61 |  3.80643 |  5.61467 |  0.850084   |
|      2 | NaivePersistence                       | RuleBased    | Azura Deluxe    |          68 |  5.14515 |  6.45826 |  0.938466   |
|      2 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          61 | 10.2708  | 12.7413  |  0.227988   |
|      2 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          61 | 10.1604  | 12.851   |  0.214632   |
|      2 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          61 | 10.5741  | 13.3685  |  0.150111   |
|      2 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          61 | 11.3987  | 14.7875  | -0.0398985  |
|      2 | SeasonalNaive7                         | RuleBased    | Side Mare Hotel |          61 | 10.2716  | 16.0239  | -0.221062   |
|      2 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          68 | 13.0638  | 18.5854  |  0.4904     |
|      2 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          68 | 12.8657  | 18.7604  |  0.480755   |
|      2 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          68 | 17.1671  | 24.1932  |  0.136476   |
|      2 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          68 | 17.6445  | 26.1584  | -0.00950255 |
|      2 | SeasonalNaive7                         | RuleBased    | Azura Deluxe    |          68 | 25.3585  | 31.815   | -0.493315   |
|      3 | NaivePersistence                       | RuleBased    | Side Mare Hotel |          68 |  1.93932 |  2.52966 |  0.630166   |
|      3 | NaivePersistence                       | RuleBased    | Azura Deluxe    |          68 |  2.48662 |  3.55175 |  0.729927   |
|      3 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          68 |  2.56665 |  3.55295 |  0.270441   |
|      3 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          68 |  2.74054 |  3.70553 |  0.206435   |
|      3 | SeasonalNaive7                         | RuleBased    | Side Mare Hotel |          68 |  3.30422 |  4.33819 | -0.0876734  |
|      3 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          68 |  3.87637 |  5.99872 |  0.229606   |
|      3 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          68 |  4.92983 |  6.09631 | -1.14791    |
|      3 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          68 |  4.06222 |  6.16223 |  0.187037   |
|      3 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          68 |  5.64935 |  6.52634 | -1.46162    |
|      3 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          68 |  4.9549  |  6.79803 |  0.0106233  |
|      3 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          68 |  5.30192 |  7.38235 | -0.166767   |
|      3 | SeasonalNaive7                         | RuleBased    | Azura Deluxe    |          68 |  6.25794 | 10.2271  | -1.23922    |
|      4 | baseline_plus_trends_walk_forward      | RandomForest | Side Mare Hotel |          68 |  2.94223 |  3.75315 |  0.976473   |
|      4 | baseline_calendar_autoreg_walk_forward | RandomForest | Side Mare Hotel |          68 |  2.8924  |  3.81661 |  0.975671   |
|      4 | NaivePersistence                       | RuleBased    | Azura Deluxe    |          68 |  3.27559 |  4.57768 |  0.969099   |
|      4 | NaivePersistence                       | RuleBased    | Side Mare Hotel |          68 |  3.00087 |  4.75069 |  0.962304   |
|      4 | baseline_plus_trends_walk_forward      | RandomForest | Azura Deluxe    |          68 |  4.81237 |  7.32278 |  0.920926   |
|      4 | baseline_calendar_autoreg_walk_forward | RandomForest | Azura Deluxe    |          68 |  4.89193 |  7.65976 |  0.913481   |
|      4 | baseline_plus_trends_walk_forward      | Ridge        | Side Mare Hotel |          68 |  9.15503 | 16.2188  |  0.560648   |
|      4 | baseline_plus_trends_walk_forward      | Ridge        | Azura Deluxe    |          68 |  9.61401 | 16.2295  |  0.611588   |
|      4 | baseline_calendar_autoreg_walk_forward | Ridge        | Azura Deluxe    |          68 | 11.9136  | 18.0199  |  0.521166   |
|      4 | baseline_calendar_autoreg_walk_forward | Ridge        | Side Mare Hotel |          68 | 11.3718  | 18.0566  |  0.455435   |
|      4 | SeasonalNaive7                         | RuleBased    | Azura Deluxe    |          68 | 10.9182  | 20.2486  |  0.395396   |
|      4 | SeasonalNaive7                         | RuleBased    | Side Mare Hotel |          68 | 10.0644  | 21.8813  |  0.200305   |

## Interpretation

- Best same-window benchmark/model: **NaivePersistence / RuleBased** with RMSE **4.068**.
- Best walk-forward benchmark/model: **NaivePersistence / RuleBased** with mean RMSE **4.170**.
- These comparisons show whether learned models and lagged Google Trends features beat simple persistence-style rules, which is a necessary check in time-series projects.