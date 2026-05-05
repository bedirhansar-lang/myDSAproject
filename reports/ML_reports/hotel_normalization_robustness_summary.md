# Hotel-wise Normalization Robustness Summary

## Purpose

This robustness pass checks whether joint conclusions remain similar after normalizing occupancy within each hotel.
The goal is to reduce the influence of between-hotel level differences when comparing pooled relationships and pooled model behavior.

## Inputs discovered automatically

- Master table: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\data\master\hotel_master_table.xlsx`
- Best lag file: `C:\Users\bedir\OneDrive\Desktop\myDSAproject\eda_outputs\best_lag_correlations.csv`

## EDA robustness setup

- Hotel-wise normalized occupancy variable: `occupancy_rate_hotel_z = (occupancy_rate - hotel_mean) / hotel_std`
- Same-day and lagged Pearson correlations are recomputed against the normalized occupancy variable.
- Compare these results against the original raw-occupancy correlations before making pooled cross-hotel claims.

### Top same-day correlations with hotel-normalized occupancy

| feature                              |   pearson_raw |   pearson_hotel_z |    abs_delta |
|:-------------------------------------|--------------:|------------------:|-------------:|
| trends_turkiye_alanya_otel           |     0.28247   |         0.260386  | -0.0220839   |
| trends_germany_alanya_hotel          |     0.239346  |         0.240763  |  0.00141674  |
| trends_turkiye_side_otel             |     0.24992   |         0.234412  | -0.0155087   |
| trends_turkiye_alanya_otel_fiyatlari |     0.21989   |         0.214309  | -0.00558069  |
| trends_netherlands_hotel_alanya      |     0.148717  |         0.146934  | -0.0017832   |
| trends_germany_hotel_turkei_side     |     0.133794  |         0.129031  | -0.00476277  |
| trends_germany_antalya_hotel         |     0.132347  |         0.127933  | -0.00441365  |
| trends_netherlands_alanya_hotel      |     0.106221  |         0.105413  | -0.000808617 |
| trends_germany_turkei_antalya_hotel  |     0.0940959 |         0.0940948 | -1.12002e-06 |
| trends_netherlands_antalya_hotel     |     0.0824122 |         0.0817363 | -0.000675903 |

### Top lagged correlations with hotel-normalized occupancy

| feature                              |   lag_days |   pearson_raw |   pearson_hotel_z |    abs_delta |
|:-------------------------------------|-----------:|--------------:|------------------:|-------------:|
| trends_turkiye_side_otel             |         28 |      0.399829 |          0.401542 |  0.00171389  |
| trends_turkiye_side_otel             |         21 |      0.366691 |          0.369612 |  0.00292095  |
| trends_turkiye_side_otel             |         14 |      0.26647  |          0.267574 |  0.0011047   |
| trends_germany_alanya_hotel          |         28 |      0.251241 |          0.257707 |  0.00646571  |
| trends_turkiye_side_otel             |          7 |      0.24974  |          0.254154 |  0.00441328  |
| trends_germany_alanya_hotel          |         21 |      0.229558 |          0.252681 |  0.023123    |
| trends_germany_alanya_hotel          |          7 |      0.255002 |          0.246128 | -0.00887447  |
| trends_turkiye_alanya_otel_fiyatlari |         14 |      0.226285 |          0.226418 |  0.000133089 |
| trends_turkiye_alanya_otel_fiyatlari |         21 |      0.229107 |          0.225786 | -0.0033217   |
| trends_turkiye_alanya_otel_fiyatlari |          7 |      0.216645 |          0.21755  |  0.000905023 |

## ML robustness setup

- Same baseline and baseline+trends feature sets are reused.
- The target is normalized within hotel using training-period hotel mean and standard deviation only, to avoid leakage.
- Predictions are also back-transformed into raw occupancy units so business-scale RMSE can still be compared.
- Baseline split date: `2025-06-28`
- Trends split date: `2025-07-22`

### Model comparison on hotel-normalized target

| dataset                                  | model        |   train_rows |   test_rows | test_start_date   | test_end_date   |   MAE_hotel_z |   RMSE_hotel_z |   R2_hotel_z |   MAE_raw_backtransformed |   RMSE_raw_backtransformed |   R2_raw_backtransformed |
|:-----------------------------------------|:-------------|-------------:|------------:|:------------------|:----------------|--------------:|---------------:|-------------:|--------------------------:|---------------------------:|-------------------------:|
| baseline_calendar_autoreg_hotel_z_target | RandomForest |          983 |         268 | 2025-06-28        | 2025-11-08      |      0.220658 |       0.366096 |     0.899314 |                   3.8638  |                    6.18916 |                 0.893124 |
| baseline_calendar_autoreg_hotel_z_target | Ridge        |          983 |         268 | 2025-06-28        | 2025-11-08      |      0.532887 |       0.919523 |     0.36481  |                   9.14148 |                   14.9241  |                 0.37857  |
| baseline_plus_trends_hotel_z_target      | RandomForest |          857 |         220 | 2025-07-22        | 2025-11-08      |      0.265465 |       0.492142 |     0.898447 |                   3.68679 |                    6.54716 |                 0.898678 |
| baseline_plus_trends_hotel_z_target      | Ridge        |          857 |         220 | 2025-07-22        | 2025-11-08      |      0.530803 |       1.06965  |     0.520268 |                   7.23489 |                   13.6191  |                 0.561579 |

### By-hotel model comparison on hotel-normalized target

| dataset                                  | model        | hotel_name      |   test_rows |   MAE_hotel_z |   RMSE_hotel_z |   R2_hotel_z |   MAE_raw_backtransformed |   RMSE_raw_backtransformed |   R2_raw_backtransformed |
|:-----------------------------------------|:-------------|:----------------|------------:|--------------:|---------------:|-------------:|--------------------------:|---------------------------:|-------------------------:|
| baseline_calendar_autoreg_hotel_z_target | RandomForest | Side Mare Hotel |         134 |      0.242996 |       0.425983 |     0.8988   |                   3.27746 |                    5.74555 |                 0.8988   |
| baseline_calendar_autoreg_hotel_z_target | RandomForest | Azura Deluxe    |         134 |      0.19832  |       0.294264 |     0.887916 |                   4.45013 |                    6.60304 |                 0.887916 |
| baseline_calendar_autoreg_hotel_z_target | Ridge        | Azura Deluxe    |         134 |      0.43659  |       0.654645 |     0.445273 |                   9.7967  |                   14.6897  |                 0.445273 |
| baseline_calendar_autoreg_hotel_z_target | Ridge        | Side Mare Hotel |         134 |      0.629183 |       1.1236   |     0.295919 |                   8.48627 |                   15.1549  |                 0.295919 |
| baseline_plus_trends_hotel_z_target      | RandomForest | Side Mare Hotel |         110 |      0.291964 |       0.579329 |     0.896484 |                   3.20827 |                    6.366   |                 0.896484 |
| baseline_plus_trends_hotel_z_target      | RandomForest | Azura Deluxe    |         110 |      0.238966 |       0.385727 |     0.900186 |                   4.16531 |                    6.72344 |                 0.900186 |
| baseline_plus_trends_hotel_z_target      | Ridge        | Azura Deluxe    |         110 |      0.435307 |       0.719009 |     0.653182 |                   7.58765 |                   12.5327  |                 0.653182 |
| baseline_plus_trends_hotel_z_target      | Ridge        | Side Mare Hotel |         110 |      0.626298 |       1.33092  |     0.453661 |                   6.88213 |                   14.6249  |                 0.453661 |

## Interpretation guide

- If the same Google Trends features remain strong after hotel-wise normalization, then the pooled signal is less likely to be driven only by level differences between hotels.
- If the baseline+trends model still outperforms the baseline model under the normalized-target setup, then the incremental value of Trends is more robust.
- If performance gains disappear, then earlier pooled gains may have been partly driven by between-hotel differences rather than within-hotel variation.