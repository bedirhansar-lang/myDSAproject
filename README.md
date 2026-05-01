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
| Strongest signal identified so far | `trends_turkiye_side_otel` at **28 days** |
| Country / keyword differences | **Meaningful differences exist**; Türkiye and Germany are often stronger than many UK features |
| ML baseline vs baseline+trends | **Lagged Google Trends improved performance**, but seasonality and past occupancy remain dominant |

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
├── scripts/
│   ├── eda_first_pass.py
│   ├── modeling_baseline.py
│   └── hotel_normalization_robustness_commented.py
│
├── reports/
│   ├── eda_summary.txt
│   ├── modeling_summary.txt
│   └── hotel_normalization_robustness_summary.md
│
├── model_outputs/
│   └── hotel_normalization_robustness/
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

This robustness layer is intended to make cross-hotel conclusions more defensible by separating:
- genuine **within-hotel movement**,
- from simple **between-hotel level differences**.

---

## Modeling Stage

After EDA, the project moved to baseline forecasting.

### Model settings compared
1. **Baseline model** using calendar features + past occupancy
2. **Baseline + lagged Google Trends** features

### Current modeling conclusion
The model with lagged Google Trends performed better than the baseline-only model, suggesting that Google Trends adds useful signal. However, **seasonality and past occupancy remain the dominant drivers** of predictive performance.

### Main modeling scripts and outputs
- Baseline modeling script: [scripts/modeling_baseline.py](scripts/modeling_baseline.py)
- Modeling summary: [reports/modeling_summary.txt](reports/modeling_summary.txt)
- Model comparison table: [reports/model_comparison.csv](reports/model_comparison.csv)
- Hotel-level comparison table: [reports/model_comparison_by_hotel.csv](reports/model_comparison_by_hotel.csv)

---

## Robustness Analysis

A separate robustness analysis was added to address the concern that the two hotels operate at different occupancy levels.

### Robustness script
- [scripts/hotel_normalization_robustness_commented.py](scripts/hotel_normalization_robustness_commented.py)

### What it does
- recomputes pooled EDA correlations using hotel-normalized occupancy,
- reruns baseline vs baseline+trends ML comparison with a hotel-normalized target,
- back-transforms predictions into raw occupancy units for interpretable RMSE comparison.

### Robustness outputs
- Summary report: [reports/hotel_normalization_robustness_summary.md](reports/hotel_normalization_robustness_summary.md)
- Output folder: [model_outputs/hotel_normalization_robustness/](model_outputs/hotel_normalization_robustness/)

---

## Interpretation

The project’s current evidence supports a cautious but useful conclusion:

- Google Trends is **not strong enough to fully explain occupancy on its own**,
- same-day search activity is **too weak** to be treated as a direct demand proxy,
- but selected **lagged Google Trends features** appear to provide meaningful incremental information,
- especially when interpreted as **early intent signals** rather than immediate booking signals.

This is consistent with the business structure of resort hotels in the region, where a substantial part of demand is mediated through **agencies and tour operators**.

---

## Next Steps

The next phase of the project focuses on:
- refined feature selection,
- stronger time-series validation,
- comparison of more interpretable ML models,
- possible inclusion of one additional external dataset,
- clearer final presentation of academic and business conclusions.

---

## Reproducibility

To reproduce the project:

1. Open the repository
2. Use the master table in `data/master/hotel_master_table.xlsx`
3. Run the scripts in `scripts/`
4. Review outputs in `reports/`, `EDA/Visualizations/`, and `model_outputs/`

---

## Project Status

**Current status:** EDA completed, baseline ML completed, hotel-wise normalization robustness analysis added, README converted into a presentation-style repository overview.
