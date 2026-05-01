"""Fair same-window comparison for hotel occupancy models.

Why this script exists:
The first-pass baseline modeling script compares
1. baseline_calendar_autoreg
2. baseline_plus_trends

but each setup drops missing rows separately before the time split. That means the
models may be evaluated on slightly different sample sets and sometimes slightly
different test windows.

This script fixes that comparison design.

Core idea:
- first build the trends-augmented dataset,
- then restrict the baseline model to the EXACT SAME rows,
- then split once by date,
- then evaluate both models on the same held-out period.

This produces a cleaner answer to the question:
"Do lagged Google Trends add predictive value beyond seasonality and past occupancy
when the comparison is made on the same sample and same test dates?"
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / 'reports' / 'ML_reports'
MODEL_DIR = REPO_ROOT / 'model_outputs' / 'fair_same_window_comparison'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def find_first_existing(candidates):
    """Return the first existing path from a list."""
    for path in candidates:
        if path.exists():
            return path
    return None


def find_master_table_path():
    """Locate the master table used across the project."""
    candidates = [
        REPO_ROOT / 'data' / 'master' / 'hotel_master_table.xlsx',
        REPO_ROOT / 'data' / 'hotel_master_table.xlsx',
        REPO_ROOT / 'hotel_master_table.xlsx',
    ]
    found = find_first_existing(candidates)
    if found is not None:
        return found
    raise FileNotFoundError('Could not find hotel_master_table.xlsx in expected locations.')


def find_best_lag_path():
    """Locate the EDA output that ranks lagged Google Trends features."""
    candidates = [
        REPO_ROOT / 'eda_outputs' / 'best_lag_correlations.csv',
        REPO_ROOT / 'reports' / 'best_lag_correlations.csv',
        REPO_ROOT / 'best_lag_correlations.csv',
    ]
    found = find_first_existing(candidates)
    if found is not None:
        return found
    matches = sorted(
        set(
            p for p in REPO_ROOT.glob('**/best_lag_correlations.csv')
            if '.git' not in p.parts
        )
    )
    if not matches:
        raise FileNotFoundError('Could not find best_lag_correlations.csv.')
    return matches[0]


# -----------------------------------------------------------------------------
# Data loading and feature engineering
# -----------------------------------------------------------------------------
def load_master_table():
    """Load the merged hotel + Google Trends table."""
    master_path = find_master_table_path()
    df = pd.read_excel(master_path, sheet_name='master_table')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['hotel_name', 'date']).reset_index(drop=True)
    return df, master_path


def add_calendar_features(df):
    """Add seasonality features used in the first-pass ML stage."""
    out = df.copy()
    out['month'] = out['date'].dt.month
    out['day_of_week'] = out['date'].dt.dayofweek
    out['week_of_year'] = out['date'].dt.isocalendar().week.astype(int)
    out['day_of_year'] = out['date'].dt.dayofyear
    out['doy_sin'] = np.sin(2 * np.pi * out['day_of_year'] / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * out['day_of_year'] / 365.25)
    return out


def add_occupancy_lags(df):
    """Add hotel-specific autoregressive occupancy features."""
    out = df.copy()
    for lag in [7, 14, 28]:
        out[f'occupancy_lag_{lag}'] = out.groupby('hotel_name')['occupancy_rate'].shift(lag)
    return out


def add_date_level_trend_lags(df, ranked_lags):
    """Create lagged trend features on unique dates, then merge them back.

    This is the correct construction because Google Trends features are shared by
    all hotels on a given date and should not be shifted directly on the hotel-date
    table.
    """
    out = df.copy()
    base_trend_cols = [
        c for c in ranked_lags['feature'].dropna().unique().tolist()
        if c in out.columns
    ]
    trend_date_df = out[['date'] + base_trend_cols].drop_duplicates('date').sort_values('date').reset_index(drop=True)

    lagged_trend_cols = []
    for _, row in ranked_lags.iterrows():
        feat = row['feature']
        lag = int(row['lag_days'])
        if feat not in trend_date_df.columns:
            continue
        new_col = f'{feat}_lag_{lag}'
        if new_col not in trend_date_df.columns:
            trend_date_df[new_col] = trend_date_df[feat].shift(lag)
        lagged_trend_cols.append(new_col)

    lagged_trend_cols = list(dict.fromkeys(lagged_trend_cols))
    if lagged_trend_cols:
        out = out.merge(trend_date_df[['date'] + lagged_trend_cols], on='date', how='left')
    return out, lagged_trend_cols


def build_feature_table():
    """Create the full aligned feature table used for fair comparison."""
    df, master_path = load_master_table()
    df = add_calendar_features(df)
    df = add_occupancy_lags(df)

    lag_path = find_best_lag_path()
    ranked_lags = pd.read_csv(lag_path).head(8).copy()
    df, lagged_trend_cols = add_date_level_trend_lags(df, ranked_lags)

    baseline_features = [
        'hotel_name', 'month', 'day_of_week', 'week_of_year', 'doy_sin', 'doy_cos',
        'occupancy_lag_7', 'occupancy_lag_14', 'occupancy_lag_28'
    ]
    trend_features = baseline_features + lagged_trend_cols

    # This is the critical fairness rule.
    # We restrict BOTH model setups to rows where the trends model is available.
    # That way both are trained and tested on the exact same observation set.
    common_df = df.dropna(subset=['occupancy_rate'] + trend_features).copy()

    return common_df, baseline_features, trend_features, lagged_trend_cols, master_path, lag_path


# -----------------------------------------------------------------------------
# Split and modeling utilities
# -----------------------------------------------------------------------------
def time_split(data, test_frac=0.2):
    """Create one temporal split based on unique dates."""
    unique_dates = np.array(sorted(pd.to_datetime(data['date'].unique())))
    split_idx = int(len(unique_dates) * (1 - test_frac))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])
    train = data[data['date'] < split_date].copy()
    test = data[data['date'] >= split_date].copy()
    return train, test, split_date


def build_preprocessor(num_cols, cat_cols):
    """Shared preprocessing for linear and tree-based benchmarks."""
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])


def evaluate_model(train_df, test_df, features, model, model_name, dataset_name):
    """Fit one model and return metrics and row-level predictions."""
    X_train = train_df[features]
    y_train = train_df['occupancy_rate']
    X_test = test_df[features]
    y_test = test_df['occupancy_rate']

    cat_cols = [c for c in features if X_train[c].dtype == 'object']
    num_cols = [c for c in features if c not in cat_cols]

    pipe = Pipeline([
        ('prep', build_preprocessor(num_cols, cat_cols)),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        'dataset': dataset_name,
        'model': model_name,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'test_start_date': test_df['date'].min().strftime('%Y-%m-%d'),
        'test_end_date': test_df['date'].max().strftime('%Y-%m-%d'),
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': math.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds),
    }

    pred_df = test_df[['date', 'hotel_name', 'occupancy_rate']].copy()
    pred_df['prediction'] = preds
    pred_df['dataset'] = dataset_name
    pred_df['model'] = model_name
    return pipe, metrics, pred_df


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def main():
    common_df, baseline_features, trend_features, lagged_trend_cols, master_path, lag_path = build_feature_table()

    # One split only, and it is shared by both setups.
    train_df, test_df, split_date = time_split(common_df)

    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('RandomForest', RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1
        ))
    ]

    results = []
    preds_all = []
    for dataset_name, features in [
        ('baseline_calendar_autoreg_same_window', baseline_features),
        ('baseline_plus_trends_same_window', trend_features),
    ]:
        for model_name, model in models:
            pipe, metrics, pred_df = evaluate_model(train_df, test_df, features, model, model_name, dataset_name)
            results.append(metrics)
            preds_all.append(pred_df)

            if model_name == 'RandomForest':
                prep = pipe.named_steps['prep']
                rf = pipe.named_steps['model']
                feature_names = prep.get_feature_names_out()
                importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                importances.to_csv(MODEL_DIR / f'{dataset_name}_{model_name.lower()}_feature_importance.csv', index=False)

    results_df = pd.DataFrame(results).sort_values(['dataset', 'RMSE'])
    preds_df = pd.concat(preds_all, ignore_index=True)

    by_hotel_rows = []
    for (dataset, model_name), g in preds_df.groupby(['dataset', 'model']):
        for hotel, hg in g.groupby('hotel_name'):
            by_hotel_rows.append({
                'dataset': dataset,
                'model': model_name,
                'hotel_name': hotel,
                'test_rows': len(hg),
                'MAE': mean_absolute_error(hg['occupancy_rate'], hg['prediction']),
                'RMSE': math.sqrt(mean_squared_error(hg['occupancy_rate'], hg['prediction'])),
                'R2': r2_score(hg['occupancy_rate'], hg['prediction']),
            })
    by_hotel_df = pd.DataFrame(by_hotel_rows).sort_values(['dataset', 'model', 'RMSE'])

    # Save outputs in a dedicated fair-comparison folder.
    results_df.to_csv(MODEL_DIR / 'model_comparison_same_window.csv', index=False)
    by_hotel_df.to_csv(MODEL_DIR / 'model_comparison_by_hotel_same_window.csv', index=False)
    preds_df.to_csv(MODEL_DIR / 'test_predictions_same_window.csv', index=False)

    # Also save report-friendly copies.
    results_df.to_csv(REPORT_DIR / 'model_comparison_same_window.csv', index=False)
    by_hotel_df.to_csv(REPORT_DIR / 'model_comparison_by_hotel_same_window.csv', index=False)

    # Plot best trends model on the shared window.
    best_row = results_df[results_df['dataset'] == 'baseline_plus_trends_same_window'].sort_values('RMSE').iloc[0]
    best_preds = preds_df[
        (preds_df['dataset'] == best_row['dataset']) &
        (preds_df['model'] == best_row['model'])
    ].copy()

    for hotel, g in best_preds.groupby('hotel_name'):
        g = g.sort_values('date')
        plt.figure(figsize=(12, 4))
        plt.plot(g['date'], g['occupancy_rate'], label='Actual')
        plt.plot(g['date'], g['prediction'], label='Predicted')
        plt.title(f'{hotel}: Actual vs Predicted Occupancy ({best_row["model"]}, fair same-window trends model)')
        plt.xlabel('Date')
        plt.ylabel('Occupancy Rate')
        plt.legend()
        plt.tight_layout()
        fname = hotel.lower().replace(' ', '_').replace('/', '_')
        plt.savefig(MODEL_DIR / f'actual_vs_pred_same_window_{fname}.png', dpi=150)
        plt.close()

    # Compact summary text for the report folder.
    base_best = results_df[results_df['dataset'] == 'baseline_calendar_autoreg_same_window'].sort_values('RMSE').iloc[0]
    trend_best = results_df[results_df['dataset'] == 'baseline_plus_trends_same_window'].sort_values('RMSE').iloc[0]
    delta = base_best['RMSE'] - trend_best['RMSE']

    summary_lines = []
    summary_lines.append('# Fair Same-Window Modeling Summary')
    summary_lines.append('')
    summary_lines.append('## Purpose')
    summary_lines.append('')
    summary_lines.append('This script compares baseline vs baseline+trends on the exact same rows and the exact same time split.')
    summary_lines.append('It addresses the fairness issue in the first-pass modeling comparison, where each setup dropped missing rows separately before splitting.')
    summary_lines.append('')
    summary_lines.append('## Inputs used')
    summary_lines.append('')
    summary_lines.append(f'- Master table: `{master_path}`')
    summary_lines.append(f'- Best lag file: `{lag_path}`')
    summary_lines.append(f'- Included lagged Trends: {", ".join(lagged_trend_cols)}')
    summary_lines.append('')
    summary_lines.append('## Shared split design')
    summary_lines.append('')
    summary_lines.append(f'- Common train rows: **{len(train_df)}**')
    summary_lines.append(f'- Common test rows: **{len(test_df)}**')
    summary_lines.append(f'- Test period starts on: **{split_date.date()}**')
    summary_lines.append('')
    summary_lines.append('## Model comparison')
    summary_lines.append('')
    summary_lines.append(results_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## By-hotel comparison')
    summary_lines.append('')
    summary_lines.append(by_hotel_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## Interpretation')
    summary_lines.append('')
    summary_lines.append(f'- Best baseline-only RMSE on the shared window: **{base_best["RMSE"]:.3f}**.')
    summary_lines.append(f'- Best baseline+trends RMSE on the shared window: **{trend_best["RMSE"]:.3f}**.')
    summary_lines.append(f'- RMSE difference (baseline - trends): **{delta:.3f}**. Positive means the trends model performed better on the same window.')
    summary_lines.append('- This result is methodologically cleaner than the first-pass comparison because both systems are evaluated on the same sample and same future period.')

    (REPORT_DIR / 'modeling_same_window_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
