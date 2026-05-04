# Hotel Occupancy Prediction with Google Trends

**DSA 210 / Undergraduate Data Science Project**  
**Region:** Antalya / Alanya, Türkiye  
**Student:** Bedirhan Sar

---

## Project Overview

This project investigates whether **Google Trends** can help explain and later predict **daily hotel occupancy rates** for resort hotels in the Antalya/Alanya region of Türkiye.

The project combines:
- daily hotel occupancy data from **Side Mare Hotel** and **Azura Deluxe**,
- tourism-related Google Trends data collected from **Germany**, **Netherlands**, **United Kingdom**, and **Türkiye**.

Because these hotels are not driven only by direct B2C demand and also rely heavily on **B2B channels** such as tour operators and travel agencies, Google Trends is not treated as a perfect booking proxy. Instead, it is evaluated as a possible **early signal of travel intent**.

### Central Question

> Can Google Trends provide useful incremental information for understanding and forecasting daily resort hotel occupancy in the Antalya/Alanya region?

---

## Main Findings

| Finding | Result |
|---|---|
| Occupancy seasonality | **Strongly present** in both hotels |
| Same-day Google Trends relationships | **Weak to moderate** |
| Lagged Google Trends relationships | **Stronger and more promising** |
| Strongest lagged signal in EDA | `trends_turkiye_side_otel` at **28 days** |
| Country / keyword differences | **Meaningful differences exist**; Türkiye and Germany are often stronger than many UK features |
| First-pass learned ML | **Lagged Google Trends improved learned-model performance**, but seasonality and past occupancy remained dominant |
| Fair same-window learned ML | **Trends still improved learned-model performance** on the same rows and same test period |
| Walk-forward learned ML | **Average improvement remained positive but modest**, with variability across folds |
| Naive benchmark comparison | **NaivePersistence outperformed all learned models** |

### Hypothesis Summary

| Hypothesis | Status |
|---|---|
| H1: Google Trends has a measurable relationship with occupancy | **Partially supported** |
| H2: Lagged Trends is more useful than same-day Trends | **Supported** |
| H3: Relationship strength depends on country and keyword | **Supported** |
| H4: Occupancy has strong seasonal structure | **Supported** |

---

## Data

### Hotel Data
- **Side Mare Hotel**
- **Azura Deluxe**
- Daily target variable: `occupancy_rate`

### Google Trends Data
Collected for 4 countries:
- **Germany**
- **Netherlands**
- **United Kingdom**
- **Türkiye**

Standardized trends schema:
- `date`
- `country`
- `keyword`
- `google_trends_score`

### Final Master Table
The merged analytical dataset has:
- **1307 rows**
- **19 columns**
- **2 hotels**
- date range: **2023-03-25 to 2025-11-08**
- **0 duplicate** `(date, hotel_name)` rows

---

## Repository Structure

```text
myDSAproject/
│
├── data/
│   └── master/
│       └── hotel_master_table.xlsx
│
├── EDA/
│   ├── EDA_detailed_report.ipynb
│   └── Visualizations/
│
├── ML/
│   └── ML_detailed_report.ipynb
│
├── scripts/
│   ├── modeling_baseline_commented.py
│   ├── hotel_normalization_robustness_commented.py
│   ├── modeling_fair_comparison_commented.py
│   ├── modeling_walk_forward_commented.py
│   └── modeling_naive_benchmarks_commented.py
│
├── model_outputs/
│   ├── baseline_ml/
│   ├── hotel_normalization_robustness/
│   ├── fair_same_window_comparison/
│   ├── walk_forward_validation/
│   └── naive_benchmark_comparison/
│
├── reports/
│   ├── EDA_Reports/
│   └── ML_reports/
│
└── README.md
```

---

## Exploratory Data Analysis

The EDA phase established the project’s core structure before moving to modeling.

Main EDA steps:
- data quality checks,
- occupancy summary by hotel,
- daily and monthly seasonality visualization,
- same-day correlation analysis,
- lagged Google Trends analysis,
- hotel-wise normalization robustness check.

### EDA Report
- Detailed notebook: [EDA/EDA_detailed_report.ipynb](EDA/EDA_detailed_report.ipynb)

### Key EDA Visualizations

![Daily Occupancy Over Time](EDA/Visualizations/occupancy_over_time.png)
*Daily occupancy over time for both hotels, showing strong seasonality and differences in volatility.*

![Monthly Average Occupancy](EDA/Visualizations/monthly_occupancy.png)
*Monthly average occupancy, making the recurring seasonal pattern easier to interpret.*

![Top Same-Day Correlations](EDA/Visualizations/top_same_day_correlations.png)
*Top same-day Pearson correlations between Google Trends features and occupancy.*

![Top Lagged Correlations](EDA/Visualizations/top_lagged_correlations.png)
*Top lagged Pearson correlations, showing that delayed search behavior is more informative than same-day search activity.*

![Best Lag Overlay](EDA/Visualizations/best_lag_overlay.png)
*Visual comparison between normalized occupancy and the strongest lagged Google Trends signal.*

### Hotel-wise Normalization Robustness Layer

Because the project pools two hotels with different average occupancy levels, an additional robustness step was added.

This robustness pass asks:
> Do the main pooled findings still hold after normalizing occupancy within each hotel?

Used normalization:

`occupancy_rate_hotel_z = (occupancy_rate - hotel_mean) / hotel_std`

Additional robustness visuals:

![Same-Day Robustness Comparison](EDA/Visualizations/hotelwise_normalization_same_day_compare_small.png)
*Same-day correlations under raw occupancy vs hotel-wise normalized occupancy.*

![Lagged Robustness Comparison](EDA/Visualizations/hotelwise_normalization_lagged_compare.png)
*Best lagged correlations under raw occupancy vs hotel-wise normalized occupancy.*

---

## Machine Learning Stage

The ML stage tests whether lagged Google Trends improves prediction beyond:
- hotel identity,
- seasonality,
- and recent occupancy history.

### ML Report
- Detailed notebook: [ML/ML_detailed_report.ipynb](ML/ML_detailed_report.ipynb)

### Learned models used
- **Ridge Regression**
- **Random Forest Regressor**

### Rule-based benchmarks used
- **NaivePersistence**
- **SeasonalNaive7**

### Validation design used across the project
1. **First-pass temporal holdout**  
2. **Hotel-wise normalization robustness**  
3. **Fair same-window comparison**  
4. **Walk-forward validation**  
5. **Naive benchmark comparison**

### Main ML scripts
- [scripts/modeling_baseline_commented.py](scripts/modeling_baseline_commented.py)
- [scripts/hotel_normalization_robustness_commented.py](scripts/hotel_normalization_robustness_commented.py)
- [scripts/modeling_fair_comparison_commented.py](scripts/modeling_fair_comparison_commented.py)
- [scripts/modeling_walk_forward_commented.py](scripts/modeling_walk_forward_commented.py)
- [scripts/modeling_naive_benchmarks_commented.py](scripts/modeling_naive_benchmarks_commented.py)

---

## First-Pass Learned ML Result

The first-pass ML comparison used a time-aware holdout split.

Main learned-model conclusion:
- best baseline-only RMSE: **approximately 5.87**
- best baseline + Trends RMSE: **approximately 4.80**

This showed that lagged Google Trends added useful predictive information **within the learned-model comparison**, although seasonality and past occupancy remained the dominant drivers.

### First-pass prediction plots

![Azura Deluxe First-Pass Prediction](model_outputs/baseline_ml/actual_vs_pred_azura_deluxe.png)
*Actual vs predicted occupancy for Azura Deluxe under the best first-pass trends-augmented learned model.*

![Side Mare Hotel First-Pass Prediction](model_outputs/baseline_ml/actual_vs_pred_side_mare_hotel.png)
*Actual vs predicted occupancy for Side Mare Hotel under the best first-pass trends-augmented learned model.*

---

## ML Robustness with Hotel-wise Normalized Target

A second ML pass used a hotel-wise normalized target to test whether the value of Trends might simply reflect level differences between the two hotels.

Main learned-model robustness conclusion:
- best baseline-only RMSE after back-transformation: **approximately 6.19**
- best baseline + Trends RMSE after back-transformation: **approximately 5.67**

This supported the view that lagged Google Trends retains some value even after controlling for hotel-specific scale differences.

### Robustness prediction plots

![Azura Deluxe Hotel-normalized Prediction](model_outputs/hotel_normalization_robustness/actual_vs_pred_hotel_z_azura_deluxe.png)
*Back-transformed prediction plot for Azura Deluxe under the hotel-normalized robustness specification.*

![Side Mare Hotel Hotel-normalized Prediction](model_outputs/hotel_normalization_robustness/actual_vs_pred_hotel_z_side_mare_hotel.png)
*Back-transformed prediction plot for Side Mare Hotel under the hotel-normalized robustness specification.*

---

## Fair Same-Window Comparison

The first-pass comparison still allowed the baseline and baseline + Trends models to be built on slightly different non-missing datasets. To fix this, a fair same-window comparison was added.

This version forces both learned model settings to use:
- the **same rows**,
- and the **same future test period**.

Main learned-model conclusion:
- best baseline-only RMSE: **4.974**
- best baseline + Trends RMSE: **4.798**

This is methodologically cleaner than the first-pass comparison and still supports a positive contribution from lagged Google Trends **inside the learned-model framework**.

### Fair same-window prediction plots

![Azura Deluxe Fair Same-Window Prediction](model_outputs/fair_same_window_comparison/actual_vs_pred_same_window_azura_deluxe.png)
*Actual vs predicted occupancy for Azura Deluxe under the fair same-window trends model.*

![Side Mare Hotel Fair Same-Window Prediction](model_outputs/fair_same_window_comparison/actual_vs_pred_same_window_side_mare_hotel.png)
*Actual vs predicted occupancy for Side Mare Hotel under the fair same-window trends model.*

---

## Walk-Forward Validation

To move beyond a single holdout, the project also uses **expanding-window walk-forward validation**.

This means the learned model is repeatedly trained on earlier dates and tested on the next future block. The goal is to check whether the Trends contribution remains useful across several future periods rather than only one final split.

Main learned-model conclusion:
- best baseline-only mean RMSE across folds: **8.166**
- best baseline + Trends mean RMSE across folds: **8.035**

So the average improvement from Trends remained **positive but modest**, while performance varied across folds.

### Walk-forward visuals

![Walk-Forward RMSE by Fold](model_outputs/walk_forward_validation/walk_forward_rmse_by_fold.png)
*Fold-by-fold RMSE comparison for baseline and trends-augmented learned models.*

![Walk-Forward Mean RMSE Summary](model_outputs/walk_forward_validation/walk_forward_mean_rmse_summary.png)
*Average RMSE across walk-forward folds for the learned models.*

---

## Naive Benchmark Comparison

The most important final ML check was to compare the learned models against simple time-series benchmark rules.

### Fair same-window benchmark result
- **NaivePersistence RMSE: 4.068**
- **SeasonalNaive7 RMSE: 7.808**
- best learned model (`baseline_plus_trends / RandomForest`) RMSE: **4.798**

### Walk-forward benchmark result
- **NaivePersistence mean RMSE: 4.170**
- **SeasonalNaive7 mean RMSE: 8.350**
- best learned model (`baseline_plus_trends / RandomForest`) mean RMSE: **8.035**

This is the most important modeling conclusion in the repository:

> Lagged Google Trends improves the learned models, but the current ML pipeline still does **not beat NaivePersistence**.

### Naive benchmark visuals

![Fair Same-Window with Naive Benchmarks](model_outputs/naive_benchmark_comparison/same_window_rmse_with_naive_benchmarks.png)
*Fair same-window RMSE comparison including rule-based benchmarks.*

![Walk-Forward RMSE with Naive Benchmarks](model_outputs/naive_benchmark_comparison/walk_forward_rmse_with_naive_benchmarks.png)
*Fold-by-fold RMSE comparison including rule-based benchmarks.*

![Walk-Forward Mean RMSE with Naive Benchmarks](model_outputs/naive_benchmark_comparison/walk_forward_mean_rmse_with_naive_benchmarks.png)
*Average RMSE across walk-forward folds including rule-based benchmarks.*

---

## Interpretation

The project’s current evidence supports a careful and more limited conclusion than a simple ML improvement story:

- Google Trends is **not strong enough to fully explain occupancy on its own**,
- same-day search activity is **too weak** to be treated as a direct demand proxy,
- selected **lagged Google Trends features** do provide useful incremental information,
- this signal survives stricter learned-model checks such as hotel-wise normalization, fair same-window comparison, and walk-forward validation,
- but the current learned models still **do not outperform NaivePersistence**,
- so daily occupancy forecasting remains strongly dominated by short-run persistence.

This is consistent with the business structure of resort hotels in the region, where a substantial part of demand is mediated through **agencies and tour operators**, and where occupancy itself is highly continuous from one day to the next.

---

## Reproducibility

To reproduce the project:

1. Open the repository.
2. Use the master table in `data/master/hotel_master_table.xlsx`.
3. Run the modeling scripts in `scripts/`.
4. Review outputs in `EDA/Visualizations/`, `model_outputs/`, and `reports/`.
5. Read the narrative reports in:
   - `EDA/EDA_detailed_report.ipynb`
   - `ML/ML_detailed_report.ipynb`

---

## Current Status

**Current status:** EDA completed, first-pass learned ML completed, hotel-normalized robustness completed, fair same-window comparison added, walk-forward validation added, naive benchmark comparison added, and final ML interpretation revised accordingly.