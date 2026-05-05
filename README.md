# Hotel Occupancy Prediction Using Google Trends

## Project Overview

This project investigates whether Google Trends search activity can help explain and predict daily hotel occupancy rates in the Antalya/Alanya tourism region of Türkiye.

The project uses daily occupancy data from two resort hotels, **Side Mare Hotel** and **Azura Deluxe**, covering the 2023, 2024, and 2025 tourism seasons. To enrich the hotel occupancy data, Google Trends data was collected for tourism-related search keywords from four countries: **Germany**, **Netherlands**, **United Kingdom**, and **Türkiye**.

The main question is:

> Can Google Trends act as an early signal of travel intent and improve daily hotel occupancy prediction?

The project follows the data science pipeline from data collection and cleaning to exploratory data analysis, statistical interpretation, machine learning, validation, benchmarking, and reporting.

## Motivation

I chose this project because I grew up around hotels and tourism through my father's work as a hotel manager. Seeing how many resources hotels spend on planning, staffing, purchasing, and daily operations made me interested in whether occupancy could be estimated more systematically using data.

Hotel occupancy is especially important in tourism regions such as Antalya and Alanya, where demand changes strongly across seasons and depends on both domestic and international travel behavior. This motivated the central idea of the project: testing whether Google search behavior can provide an early signal about future hotel demand.

The project does not claim to represent every hotel in Antalya. Instead, it uses two real hotels from the Antalya region as a focused starting point for testing whether Google Trends can provide useful explanatory or predictive information.

## Dataset

### Hotel occupancy data

The hotel occupancy data consists of daily occupancy rates from:

- **Side Mare Hotel**
- **Azura Deluxe**

The data covers three tourism seasons:

- **2023**
- **2024**
- **2025**

The first hotel dataset was obtained by requesting internal occupancy records through my father's hotel. To enrich the project and avoid relying on only one hotel, I contacted another hotel and obtained a second daily occupancy dataset. The raw hotel data was cleaned and standardized into a common format with one row per hotel-date observation.

Main standardized columns:

- `date`
- `hotel_name`
- `occupancy_rate`

### Google Trends data

Google Trends data was collected manually through the **Google Trends web interface** and downloaded as CSV files. The search region and language context were selected separately for four countries that are relevant to Antalya/Alanya tourism demand:

- Germany
- Netherlands
- United Kingdom
- Türkiye

Several tourism-related keywords were tested in the Antalya holiday context. The keywords with the most usable data coverage were selected for the final analysis. These Trends files were cleaned, standardized, reshaped, and merged with the hotel occupancy data by date.

## Repository Structure

```text
myDSAproject/
│
├── data/
│   ├── cleaned/
│   ├── master/
│   └── raw/
│
├── EDA/
│   ├── EDA_detailed_report.ipynb
│   └── Figures/
│
├── ML/
│   ├── ML_detailed_report.ipynb
│   ├── Figures/
│   │   └── Naive_Benchmark/
│   └── FIGURE_REPRODUCIBILITY.md
│
├── reports/
│   ├── EDA_reports/
│   │   └── best_lag_correlations.csv
│   └── ML_reports/
│
├── scripts/
│   ├── run_ml_pipeline.py
│   ├── modeling_baseline_commented.py
│   ├── hotel_normalization_robustness_commented.py
│   ├── modeling_fair_comparison_commented.py
│   ├── modeling_walk_forward_commented.py
│   ├── modeling_naive_benchmarks_commented.py
│   └── sync_ml_report_figures.py
│
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data preparation

The hotel datasets were cleaned into a consistent daily format. Google Trends CSV files were cleaned into a common date-country-keyword structure, then reshaped into wide format so each Trends keyword-country combination could be used as a separate feature.

The final master table contains one row per hotel and date, including:

- hotel identity
- daily occupancy rate
- calendar features
- Google Trends variables

### 2. Exploratory data analysis

The EDA stage examined:

- seasonal patterns in hotel occupancy
- hotel-level differences
- same-day Google Trends relationships
- lagged Google Trends relationships
- country and keyword differences

The EDA results showed that same-day Google Trends relationships were generally weak to moderate, while selected lagged Trends signals were more promising. This supported the decision to test 7, 14, 21, and 28-day lagged Google Trends features in the machine learning stage.

### 3. Feature engineering

The ML stage used:

- hotel identity
- calendar features such as month, day of week, and week of year
- cyclical seasonality features
- occupancy lag features
- selected lagged Google Trends features

The selected Trends features were taken from the EDA stage by ranking feature-lag combinations according to their absolute Pearson correlation with occupancy rate. These ranked feature-lag combinations are stored in:

```text
reports/EDA_reports/best_lag_correlations.csv
```

An important correction was made during the project: Google Trends features are date-level signals shared across hotels, so their lagged versions must be created on a unique date table before being merged back into the hotel-date modeling table. This avoids incorrect lag alignment across hotel rows.

### 4. Machine learning models

Two learned models were compared:

- **Ridge Regression**: a regularized linear benchmark
- **Random Forest Regressor**: a non-linear model that can capture interactions between seasonality, hotel identity, lagged occupancy, and Google Trends

Two main feature settings were evaluated:

1. **Baseline model**
   - hotel identity
   - calendar and seasonality features
   - occupancy lags

2. **Baseline + Google Trends model**
   - all baseline features
   - selected lagged Google Trends features

### 5. Validation and benchmarking

The project avoids random train-test splitting because the observations are time-ordered. Instead, the ML stage uses time-aware validation strategies:

- chronological holdout validation
- fair same-window validation
- expanding-window walk-forward validation
- naive benchmark comparison

The naive benchmark comparison includes:

- **NaivePersistence**: predicts today's occupancy using yesterday's occupancy for the same hotel
- **SeasonalNaive7**: predicts today's occupancy using occupancy from seven days earlier

This benchmark step is important because a learned model should not only improve over another learned model; it should also be compared against simple forecasting rules.

## Key Findings

### EDA findings

- Hotel occupancy is strongly seasonal.
- Same-day Google Trends relationships are generally not strong enough to serve as direct demand proxies.
- Lagged Google Trends features are more useful than same-day Trends values.
- Some Türkiye and Germany keyword combinations provided stronger signals than many United Kingdom keywords.
- The strongest Trends relationships appeared around selected 21-day and 28-day lags.

### ML findings

The learned-model comparisons showed that adding selected lagged Google Trends features improved prediction compared with a baseline model using only hotel identity, seasonality, and past occupancy.

However, the strongest simple benchmark remained **NaivePersistence**. This means that daily hotel occupancy is highly persistence-driven: yesterday's occupancy is a very strong short-term predictor of today's occupancy.

The main ML conclusion is therefore careful and limited:

> Lagged Google Trends features add useful supporting signal to learned models, but they do not yet outperform simple short-term persistence. Google Trends should be interpreted as an early supporting indicator of travel intent rather than a standalone forecasting signal.

## Reports

Detailed project reports are available in the following notebooks:

- `EDA/EDA_detailed_report.ipynb`
- `ML/ML_detailed_report.ipynb`

The ML report includes the full modeling logic, validation comparisons, benchmark results, figures, interpretation, limitations, and a short question-answer section for likely evaluation questions.

## Reproducing the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/bedirhansar-lang/myDSAproject.git
cd myDSAproject
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the complete ML pipeline

```bash
python scripts/run_ml_pipeline.py
```

This command runs the ML scripts in the correct order, regenerates model output tables, recreates figures under `model_outputs/`, and syncs report-ready figure copies into `ML/Figures/`.

### 4. Open the reports

```bash
python -m notebook EDA/EDA_detailed_report.ipynb
python -m notebook ML/ML_detailed_report.ipynb
```

If Jupyter is not available as a direct command, use:

```bash
python -m notebook
```

and then open the notebooks through the browser interface.

## Requirements

The main Python dependencies are listed in:

```text
requirements.txt
```

Core libraries include:

- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- openpyxl
- jupyter

## Limitations and Future Work

This project has several limitations:

- The dataset includes two hotels, so the results should not be generalized to all Antalya hotels.
- Hotel demand is affected by tour operators, travel agencies, pricing, flights, school holidays, macroeconomic conditions, and other variables that are not fully captured in the current dataset.
- Google Trends data is normalized search interest, not direct booking or reservation data.
- Same-day Google Trends values are weak direct demand proxies.
- The current learned models do not outperform the strongest naive persistence benchmark.

Possible future extensions include:

- adding price, booking, flight, or weather data
- including more hotels from different Antalya subregions
- testing longer forecasting horizons where yesterday's occupancy is less dominant
- using more advanced time-series models
- evaluating whether Google Trends is more useful for weekly or monthly planning than for next-day prediction

## Academic Integrity and AI Usage

AI tools were used in this project as support for planning, code commenting, debugging, repository organization, documentation writing, and methodological discussion. In particular, AI assistance was used to help structure the EDA-to-ML workflow, organize the repository folders, improve the README and notebook reports, detect issues in lagged Google Trends feature construction, and design additional analysis steps such as hotel-wise normalization robustness, fair same-window comparison, walk-forward validation, and naive benchmark comparison. AI was also used to help interpret model outputs more clearly and present the findings in an academically transparent way. Final decisions about the project design, implementation, result selection, and interpretation were made by the student.
