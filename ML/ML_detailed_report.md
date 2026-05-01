# Machine Learning Report

## Hotel Occupancy Prediction for Antalya/Alanya Resort Hotels Using Google Trends

**Prepared by:** Bedirhan Sar  
**Project:** Hotel Occupancy Analysis and Prediction

---

## Purpose of the ML Stage

The EDA phase showed that:
- occupancy is strongly seasonal,
- same-day Google Trends relationships are generally weak,
- several lagged Google Trends features are more promising,
- and the strongest signals come mainly from selected Türkiye and Germany keywords.

The purpose of the ML stage is therefore not to replace the EDA logic, but to test a focused predictive question:

> Do lagged Google Trends features improve hotel occupancy prediction beyond hotel identity, calendar seasonality, and recent occupancy history?

---

## Modeling Strategy Used So Far

Two modeling settings were compared.

### 1. Baseline model
Feature set:
- hotel identity,
- calendar features,
- cyclical seasonality encoding,
- occupancy lags at 7, 14, and 28 days.

This setup captures the main internal structure of the problem: hotel-specific level differences, seasonality, and autoregressive behavior.

### 2. Baseline + lagged Google Trends model
This setup includes everything in the baseline model, plus selected lagged Google Trends variables chosen from the EDA stage.

The intention is to test whether Trends adds **incremental signal** beyond what occupancy history and seasonality already explain.

---

## Feature Engineering

### Calendar and seasonality features
The following features were used to capture recurring temporal structure:
- `month`
- `day_of_week`
- `week_of_year`
- `doy_sin`
- `doy_cos`

### Autoregressive occupancy features
Recent hotel behavior was represented with:
- `occupancy_lag_7`
- `occupancy_lag_14`
- `occupancy_lag_28`

### Lagged Google Trends features
The strongest feature-lag combinations from the earlier EDA pass were used as a compact first-pass Trends feature set.

Important correction applied in the latest version:

Because Google Trends variables are **date-level shared signals**, their lagged versions must be created on a **unique date table first** and then merged back by date. They should not be shifted directly on the hotel-date modeling table.

This alignment issue has now been corrected in:
- `scripts/modeling_baseline_commented.py`
- `scripts/hotel_normalization_robustness_commented.py`

---

## Models Compared

Two models were used in the first-pass ML comparison:

### Ridge Regression
A regularized linear benchmark.

### Random Forest Regressor
A stronger non-linear benchmark that can model interactions and non-linear relationships without requiring explicit feature transformations.

---

## Validation Design

A **time-aware split** was used instead of random train/test splitting.

This means:
- earlier dates were used for training,
- later dates were used for testing.

This is more appropriate than random splitting because the project is forecasting-like and must respect temporal ordering.

---

## Current Baseline ML Findings

After correcting lagged Google Trends alignment and rerunning the first-pass pipeline, the main result remained consistent:

- the best baseline-only model was weaker,
- the best baseline + lagged Trends model performed better,
- and the best overall configuration remained **Random Forest with lagged Google Trends**.

### Updated pooled result
- Best baseline-only RMSE: **approximately 5.87**
- Best baseline + Trends RMSE: **approximately 4.80**

This means that lagged Google Trends still appears to add useful predictive information after the lag-construction bug was corrected.

---

## Hotel-Level Prediction Plots

These figures show actual vs predicted occupancy for the best non-robustness trends-augmented model.

### Azura Deluxe
![Azura Deluxe prediction](../model_outputs/baseline_ml/actual_vs_pred_azura_deluxe.png)

### Side Mare Hotel
![Side Mare Hotel prediction](../model_outputs/baseline_ml/actual_vs_pred_side_mare_hotel.png)

---

## Hotel-wise Normalization Robustness in the ML Stage

The EDA stage already tested whether pooled correlation findings survive hotel-wise normalization. The ML stage extends that idea by asking:

> If the target is normalized within each hotel, does the incremental value of lagged Google Trends remain visible?

This matters because the two hotels do not operate at exactly the same occupancy level or variability scale.

### Normalized target used in robustness modeling

`target_hotel_z = (occupancy_rate - train_hotel_mean) / train_hotel_std`

Important leakage rule:
- hotel mean and standard deviation were computed from the **train period only**,
- then applied to both train and test.

This keeps the normalization leakage-safe.

### Robustness result
After rerunning the corrected robustness pipeline:
- the best hotel-normalized baseline-only model remained weaker,
- the best hotel-normalized baseline + Trends model still performed better,
- and the best model again remained **Random Forest with lagged Google Trends**.

### Updated robustness result (raw scale after back-transformation)
- Best baseline-only RMSE: **approximately 6.19**
- Best baseline + Trends RMSE: **approximately 5.67**

This is important because it suggests that the value of lagged Google Trends is **not only an artifact of pooling two hotels with different average occupancy levels**.

---

## Hotel-normalized Prediction Plots

These figures show actual vs predicted occupancy for the best hotel-normalized robustness model, back-transformed to raw occupancy scale.

### Azura Deluxe
![Azura Deluxe normalized-target robustness prediction](../model_outputs/hotel_normalization_robustness/actual_vs_pred_hotel_z_azura_deluxe.png)

### Side Mare Hotel
![Side Mare Hotel normalized-target robustness prediction](../model_outputs/hotel_normalization_robustness/actual_vs_pred_hotel_z_side_mare_hotel.png)

---

## What Has Been Done Correctly

At the current project stage, the ML pipeline already includes several strong design choices:

- time-aware train/test splitting,
- separate comparison of baseline vs baseline + Trends,
- hotel-level lag creation for occupancy,
- pooled and hotel-level evaluation,
- robustness analysis with hotel-wise normalized target,
- train-only normalization for the robustness target,
- back-transformed RMSE reporting for business interpretability.

---

## Current Limitations of the ML Stage

Although the first-pass ML stage is meaningful, it is not yet the final modeling design.

### 1. Single holdout split
The current pipeline still uses a single temporal holdout split rather than repeated walk-forward validation.

### 2. Comparison fairness still needs tightening
The baseline and baseline + Trends models are currently built on slightly different non-missing datasets. The next version should enforce the same comparison window more strictly.

### 3. Benchmark set can be improved
The current benchmark set includes Ridge and Random Forest. Additional baselines such as naive persistence or a simpler statistical baseline would make the comparison stronger.

### 4. Feature refinement is still incomplete
The current Trends feature set is a compact first-pass selection. A more systematic feature refinement stage is still needed.

---

## Interpretation

The ML evidence so far supports a careful but useful conclusion:

- **seasonality and recent occupancy remain the dominant drivers**,
- but **lagged Google Trends still adds incremental predictive value**,
- and this conclusion remains visible even after hotel-wise normalization.

So the current ML result is aligned with the EDA logic:
- Google Trends is not a strong same-day demand proxy,
- but selected lagged search signals may act as an **early supporting indicator**.

---

## Recommended Next ML Steps

The next clean steps for the project are:

1. enforce an identical comparison window for baseline and baseline + Trends,  
2. replace the single holdout with rolling or expanding-window validation,  
3. compare stronger but still interpretable benchmark models,  
4. refine feature selection using only properly aligned lagged features,  
5. summarize final conclusions in a more presentation-ready ML results section.

---

## Related Files

Main scripts:
- `scripts/modeling_baseline_commented.py`
- `scripts/hotel_normalization_robustness_commented.py`

Main tabular outputs:
- `reports/model_comparison.csv`
- `reports/model_comparison_by_hotel.csv`
- `reports/modeling_summary.txt`
- `model_outputs/hotel_normalization_robustness/model_comparison_hotel_normalized_target.csv`
- `model_outputs/hotel_normalization_robustness/model_comparison_by_hotel_hotel_normalized_target.csv`

---

## Current Status

**Status:** First-pass ML completed, lagged Google Trends alignment bug corrected, corrected outputs regenerated, ML robustness section separated conceptually from the EDA report.