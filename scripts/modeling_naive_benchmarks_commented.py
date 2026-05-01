"""Naive benchmark comparison for hotel occupancy prediction.

Why this script exists:
The ML stage currently compares learned models such as Ridge and Random Forest.
That is useful, but time-series projects should also include simple rule-based
benchmarks so we can judge whether the learned models are truly adding value.

This script adds two benchmarks:
1. Naive persistence
   - predict today's occupancy as yesterday's occupancy for the same hotel
2. Seasonal naive (7-day)
   - predict today's occupancy as the occupancy from 7 days earlier for the same hotel

The script evaluates these rule-based baselines against the learned baseline and
baseline+trends models in two settings:
- fair same-window comparison
- walk-forward validation

This gives a more honest answer to the question:
"Are the learned models, and especially the lagged Google Trends models,
better than simple time-series benchmark rules?"
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
MODEL_DIR = REPO_ROOT / 'model_outputs' / 'naive_benchmark_comparison'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# File discovery helpers
# -----------------------------------------------------------------------------
def find_first_existing(candidates):
    """Return the first existing path from a candidate list."""
    for path in candidates:
        if path.exists():
            return path
    return None


def find_master_table_path():
    """Locate the project master table."""
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
    """Locate the EDA file that ranks lagged Google Trends features."""
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
    """Load and sort the hotel master table."""
    master_path = find_master_table_path()
    df = pd.read_excel(master_path, sheet_name='master_table')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['hotel_name', 'date']).reset_index(drop=True)
    return df, master_path


def add_calendar_features(df):
    """Add calendar features that capture seasonality."""
    out = df.copy()
    out['month'] = out['date'].dt.month
    out['day_of_week'] = out['date'].dt.dayofweek
    out['week_of_year'] = out['date'].dt.isocalendar().week.astype(int)
    out['day_of_year'] = out['date'].dt.dayofyear
    out['doy_sin'] = np.sin(2 * np.pi * out['day_of_year'] / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * out['day_of_year'] / 365.25)
    return out


def add_occupancy_lags(df):
    """Add within-hotel occupancy lags.

    The 1-day lag is required for the naive persistence benchmark.
    The 7-day lag is used both in the learned baseline and in the seasonal naive benchmark.
    """
    out = df.copy()
    for lag in [1, 7, 14, 28]:
        out[f'occupancy_lag_{lag}'] = out.groupby('hotel_name')['occupancy_rate'].shift(lag)
    return out


def add_date_level_trend_lags(df, ranked_lags):
    """Construct lagged Google Trends features at the date level, then merge back."""
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


def build_common_feature_table():
    """Build one aligned table shared by all learned models and naive benchmarks.

    Fairness rule:
    - the learned baseline model,
    - the learned baseline+trends model,
    - naive persistence,
    - seasonal naive
    should all be evaluated on the exact same rows.

    Therefore we keep only rows where:
    - the target exists,
    - the trends features exist,
    - occupancy_lag_1 exists,
    - occupancy_lag_7 exists.
    """
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

    common_df = df.dropna(
        subset=['occupancy_rate', 'occupancy_lag_1', 'occupancy_lag_7'] + trend_features
    ).copy()
    return common_df, baseline_features, trend_features, lagged_trend_cols, master_path, lag_path


# -----------------------------------------------------------------------------
# Split logic
# -----------------------------------------------------------------------------
def time_split(data, test_frac=0.2):
    """Create one fair temporal holdout split."""
    unique_dates = np.array(sorted(pd.to_datetime(data['date'].unique())))
    split_idx = int(len(unique_dates) * (1 - test_frac))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])
    train = data[data['date'] < split_date].copy()
    test = data[data['date'] >= split_date].copy()
    return train, test, split_date


def make_walk_forward_folds(data, n_folds=4, min_train_frac=0.5):
    """Create expanding-window walk-forward folds using unique dates."""
    unique_dates = np.array(sorted(pd.to_datetime(data['date'].unique())))
    n_dates = len(unique_dates)
    if n_dates < 40:
        raise ValueError('Not enough unique dates for walk-forward validation.')

    initial_train_size = max(int(n_dates * min_train_frac), 30)
    remaining = n_dates - initial_train_size
    if remaining < n_folds:
        raise ValueError('Not enough remaining dates to create the requested number of folds.')

    fold_sizes = [remaining // n_folds] * n_folds
    for i in range(remaining % n_folds):
        fold_sizes[i] += 1

    folds = []
    train_end = initial_train_size
    test_start = initial_train_size
    for fold_idx, fold_size in enumerate(fold_sizes, start=1):
        test_end = test_start + fold_size
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[test_start:test_end]
        folds.append({
            'fold': fold_idx,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'train_start_date': pd.Timestamp(train_dates[0]),
            'train_end_date': pd.Timestamp(train_dates[-1]),
            'test_start_date': pd.Timestamp(test_dates[0]),
            'test_end_date': pd.Timestamp(test_dates[-1]),
        })
        train_end = test_end
        test_start = test_end
    return folds


# -----------------------------------------------------------------------------
# Modeling utilities
# -----------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols):
    """Shared preprocessing for learned models."""
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
    """Fit one learned model and return metrics and predictions."""
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
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': math.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds),
    }

    pred_df = test_df[['date', 'hotel_name', 'occupancy_rate']].copy()
    pred_df['prediction'] = preds
    pred_df['dataset'] = dataset_name
    pred_df['model'] = model_name
    return pipe, metrics, pred_df


def evaluate_rule_based(test_df, pred_col, benchmark_name):
    """Evaluate a simple benchmark that predicts from an existing lag column."""
    preds = test_df[pred_col].values
    y_test = test_df['occupancy_rate'].values

    metrics = {
        'dataset': benchmark_name,
        'model': 'RuleBased',
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': math.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds),
    }

    pred_df = test_df[['date', 'hotel_name', 'occupancy_rate']].copy()
    pred_df['prediction'] = preds
    pred_df['dataset'] = benchmark_name
    pred_df['model'] = 'RuleBased'
    return metrics, pred_df


# -----------------------------------------------------------------------------
# Same-window benchmark comparison
# -----------------------------------------------------------------------------
def run_same_window(common_df, baseline_features, trend_features):
    """Run a fair same-window comparison with rule-based benchmarks included."""
    train_df, test_df, split_date = time_split(common_df)

    results = []
    preds_all = []
    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('RandomForest', RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1
        )),
    ]

    for dataset_name, features in [
        ('baseline_calendar_autoreg_same_window', baseline_features),
        ('baseline_plus_trends_same_window', trend_features),
    ]:
        for model_name, model in models:
            _, metrics, pred_df = evaluate_model(train_df, test_df, features, model, model_name, dataset_name)
            metrics['train_rows'] = len(train_df)
            metrics['test_rows'] = len(test_df)
            metrics['test_start_date'] = test_df['date'].min().strftime('%Y-%m-%d')
            metrics['test_end_date'] = test_df['date'].max().strftime('%Y-%m-%d')
            results.append(metrics)
            preds_all.append(pred_df)

    for benchmark_name, pred_col in [
        ('NaivePersistence', 'occupancy_lag_1'),
        ('SeasonalNaive7', 'occupancy_lag_7'),
    ]:
        metrics, pred_df = evaluate_rule_based(test_df, pred_col, benchmark_name)
        metrics['train_rows'] = len(train_df)
        metrics['test_rows'] = len(test_df)
        metrics['test_start_date'] = test_df['date'].min().strftime('%Y-%m-%d')
        metrics['test_end_date'] = test_df['date'].max().strftime('%Y-%m-%d')
        results.append(metrics)
        preds_all.append(pred_df)

    results_df = pd.DataFrame(results).sort_values('RMSE')
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
    by_hotel_df = pd.DataFrame(by_hotel_rows).sort_values(['RMSE', 'dataset'])

    # Save outputs.
    results_df.to_csv(MODEL_DIR / 'same_window_with_naive_benchmarks.csv', index=False)
    by_hotel_df.to_csv(MODEL_DIR / 'same_window_with_naive_benchmarks_by_hotel.csv', index=False)
    preds_df.to_csv(MODEL_DIR / 'same_window_with_naive_benchmarks_predictions.csv', index=False)

    results_df.to_csv(REPORT_DIR / 'same_window_with_naive_benchmarks.csv', index=False)
    by_hotel_df.to_csv(REPORT_DIR / 'same_window_with_naive_benchmarks_by_hotel.csv', index=False)

    # Plot same-window RMSE ranking.
    plt.figure(figsize=(8, 5))
    labels = results_df['dataset'] + ' | ' + results_df['model']
    plt.bar(labels, results_df['RMSE'])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('RMSE')
    plt.title('Fair same-window comparison with naive benchmarks')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'same_window_rmse_with_naive_benchmarks.png', dpi=150)
    plt.close()

    return results_df, by_hotel_df, preds_df, split_date


# -----------------------------------------------------------------------------
# Walk-forward benchmark comparison
# -----------------------------------------------------------------------------
def run_walk_forward(common_df, baseline_features, trend_features):
    """Run walk-forward validation with rule-based benchmarks included."""
    folds = make_walk_forward_folds(common_df, n_folds=4, min_train_frac=0.5)

    results = []
    preds_all = []
    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('RandomForest', RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1
        )),
    ]

    for fold_info in folds:
        fold_id = fold_info['fold']
        train_df = common_df[common_df['date'].isin(fold_info['train_dates'])].copy()
        test_df = common_df[common_df['date'].isin(fold_info['test_dates'])].copy()

        for dataset_name, features in [
            ('baseline_calendar_autoreg_walk_forward', baseline_features),
            ('baseline_plus_trends_walk_forward', trend_features),
        ]:
            for model_name, model in models:
                _, metrics, pred_df = evaluate_model(train_df, test_df, features, model, model_name, dataset_name)
                metrics['fold'] = fold_id
                metrics['train_rows'] = len(train_df)
                metrics['test_rows'] = len(test_df)
                metrics['test_start_date'] = test_df['date'].min().strftime('%Y-%m-%d')
                metrics['test_end_date'] = test_df['date'].max().strftime('%Y-%m-%d')
                results.append(metrics)
                preds_all.append(pred_df.assign(fold=fold_id))

        for benchmark_name, pred_col in [
            ('NaivePersistence', 'occupancy_lag_1'),
            ('SeasonalNaive7', 'occupancy_lag_7'),
        ]:
            metrics, pred_df = evaluate_rule_based(test_df, pred_col, benchmark_name)
            metrics['fold'] = fold_id
            metrics['train_rows'] = len(train_df)
            metrics['test_rows'] = len(test_df)
            metrics['test_start_date'] = test_df['date'].min().strftime('%Y-%m-%d')
            metrics['test_end_date'] = test_df['date'].max().strftime('%Y-%m-%d')
            results.append(metrics)
            preds_all.append(pred_df.assign(fold=fold_id))

    results_df = pd.DataFrame(results).sort_values(['fold', 'RMSE'])
    preds_df = pd.concat(preds_all, ignore_index=True)

    summary_df = (
        results_df
        .groupby(['dataset', 'model'], as_index=False)
        .agg(
            folds=('fold', 'nunique'),
            mean_MAE=('MAE', 'mean'),
            std_MAE=('MAE', 'std'),
            mean_RMSE=('RMSE', 'mean'),
            std_RMSE=('RMSE', 'std'),
            mean_R2=('R2', 'mean'),
            std_R2=('R2', 'std'),
        )
        .sort_values('mean_RMSE')
    )

    by_hotel_rows = []
    for (fold_id, dataset, model_name), g in preds_df.groupby(['fold', 'dataset', 'model']):
        for hotel, hg in g.groupby('hotel_name'):
            by_hotel_rows.append({
                'fold': fold_id,
                'dataset': dataset,
                'model': model_name,
                'hotel_name': hotel,
                'test_rows': len(hg),
                'MAE': mean_absolute_error(hg['occupancy_rate'], hg['prediction']),
                'RMSE': math.sqrt(mean_squared_error(hg['occupancy_rate'], hg['prediction'])),
                'R2': r2_score(hg['occupancy_rate'], hg['prediction']),
            })
    by_hotel_df = pd.DataFrame(by_hotel_rows).sort_values(['fold', 'RMSE'])

    # Save outputs.
    results_df.to_csv(MODEL_DIR / 'walk_forward_with_naive_benchmarks_fold_results.csv', index=False)
    summary_df.to_csv(MODEL_DIR / 'walk_forward_with_naive_benchmarks_summary.csv', index=False)
    by_hotel_df.to_csv(MODEL_DIR / 'walk_forward_with_naive_benchmarks_by_hotel.csv', index=False)
    preds_df.to_csv(MODEL_DIR / 'walk_forward_with_naive_benchmarks_predictions.csv', index=False)

    results_df.to_csv(REPORT_DIR / 'walk_forward_with_naive_benchmarks_fold_results.csv', index=False)
    summary_df.to_csv(REPORT_DIR / 'walk_forward_with_naive_benchmarks_summary.csv', index=False)
    by_hotel_df.to_csv(REPORT_DIR / 'walk_forward_with_naive_benchmarks_by_hotel.csv', index=False)

    # Plot fold-by-fold RMSE.
    pivot_rmse = results_df.pivot_table(index='fold', columns=['dataset', 'model'], values='RMSE')
    plt.figure(figsize=(10, 5))
    for col in pivot_rmse.columns:
        plt.plot(pivot_rmse.index, pivot_rmse[col], marker='o', label=' | '.join(col))
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('Walk-forward RMSE with naive benchmarks')
    plt.xticks(sorted(results_df['fold'].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'walk_forward_rmse_with_naive_benchmarks.png', dpi=150)
    plt.close()

    # Plot mean RMSE summary.
    plt.figure(figsize=(8, 5))
    labels = summary_df['dataset'] + ' | ' + summary_df['model']
    plt.bar(labels, summary_df['mean_RMSE'])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Mean RMSE across folds')
    plt.title('Walk-forward mean RMSE with naive benchmarks')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'walk_forward_mean_rmse_with_naive_benchmarks.png', dpi=150)
    plt.close()

    return results_df, summary_df, by_hotel_df


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def main():
    common_df, baseline_features, trend_features, lagged_trend_cols, master_path, lag_path = build_common_feature_table()

    same_window_df, same_window_by_hotel_df, _, split_date = run_same_window(
        common_df, baseline_features, trend_features
    )
    walk_forward_fold_df, walk_forward_summary_df, walk_forward_by_hotel_df = run_walk_forward(
        common_df, baseline_features, trend_features
    )

    # Write a compact markdown summary.
    best_same_window = same_window_df.sort_values('RMSE').iloc[0]
    best_walk_forward = walk_forward_summary_df.sort_values('mean_RMSE').iloc[0]

    summary_lines = []
    summary_lines.append('# Naive Benchmark Comparison Summary')
    summary_lines.append('')
    summary_lines.append('## Purpose')
    summary_lines.append('')
    summary_lines.append('This script compares the learned baseline and baseline+trends models against two simple rule-based time-series benchmarks: naive persistence and seasonal naive (7-day).')
    summary_lines.append('')
    summary_lines.append('## Inputs used')
    summary_lines.append('')
    summary_lines.append(f'- Master table: `{master_path}`')
    summary_lines.append(f'- Best lag file: `{lag_path}`')
    summary_lines.append(f'- Included lagged Trends: {", ".join(lagged_trend_cols)}')
    summary_lines.append('')
    summary_lines.append('## Fair same-window comparison')
    summary_lines.append('')
    summary_lines.append(f'- Shared test period starts on: **{split_date.date()}**')
    summary_lines.append(same_window_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('### By-hotel same-window comparison')
    summary_lines.append('')
    summary_lines.append(same_window_by_hotel_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## Walk-forward comparison')
    summary_lines.append('')
    summary_lines.append(walk_forward_summary_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('### By-hotel walk-forward comparison')
    summary_lines.append('')
    summary_lines.append(walk_forward_by_hotel_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## Interpretation')
    summary_lines.append('')
    summary_lines.append(f"- Best same-window benchmark/model: **{best_same_window['dataset']} / {best_same_window['model']}** with RMSE **{best_same_window['RMSE']:.3f}**.")
    summary_lines.append(f"- Best walk-forward benchmark/model: **{best_walk_forward['dataset']} / {best_walk_forward['model']}** with mean RMSE **{best_walk_forward['mean_RMSE']:.3f}**.")
    summary_lines.append('- These comparisons show whether learned models and lagged Google Trends features beat simple persistence-style rules, which is a necessary check in time-series projects.')

    (REPORT_DIR / 'naive_benchmark_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
