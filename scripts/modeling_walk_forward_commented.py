"""Walk-forward validation for hotel occupancy prediction.

Why this script exists:
The first-pass ML stage used a single temporal holdout. That is a good start,
but it is not strong enough for a final time-series modeling claim.

This script upgrades the evaluation design by using expanding-window
walk-forward validation on a fair same-sample comparison.

Core design choices:
- build the trends-augmented feature table correctly,
- restrict both baseline and baseline+trends models to the same rows,
- create multiple chronological folds,
- train on earlier dates and test on the next time block,
- compare baseline vs baseline+trends across folds rather than on only one split.

This gives a cleaner answer to the question:
"Does lagged Google Trends improve predictive performance consistently across
multiple future periods, not just in one final holdout window?"
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
MODEL_DIR = REPO_ROOT / 'model_outputs' / 'walk_forward_validation'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# File discovery helpers
# -----------------------------------------------------------------------------
def find_first_existing(candidates):
    """Return the first path that exists from a candidate list."""
    for path in candidates:
        if path.exists():
            return path
    return None


def find_master_table_path():
    """Locate the master table file used across the project."""
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
    """Locate the EDA file that ranks lagged trend features."""
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
    """Add within-hotel occupancy lags."""
    out = df.copy()
    for lag in [7, 14, 28]:
        out[f'occupancy_lag_{lag}'] = out.groupby('hotel_name')['occupancy_rate'].shift(lag)
    return out


def add_date_level_trend_lags(df, ranked_lags):
    """Construct lagged Google Trends features at the date level, then merge back.

    This is the correct method because the trends series are shared by all hotels on
    the same date and must not be shifted directly on the hotel-date table.
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


def build_common_feature_table():
    """Build one aligned feature table shared by both model settings.

    Fairness rule:
    both baseline and trends models must be evaluated on the exact same rows.
    Therefore we keep only rows where the full trends feature set is available.
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

    common_df = df.dropna(subset=['occupancy_rate'] + trend_features).copy()
    return common_df, baseline_features, trend_features, lagged_trend_cols, master_path, lag_path


# -----------------------------------------------------------------------------
# Walk-forward split logic
# -----------------------------------------------------------------------------
def make_walk_forward_folds(data, n_folds=4, min_train_frac=0.5):
    """Create expanding-window walk-forward folds using unique dates.

    Parameters:
    - n_folds: number of future test blocks
    - min_train_frac: minimum fraction of dates reserved for the first train block

    Output:
    A list of dictionaries with train/test date ranges.
    """
    unique_dates = np.array(sorted(pd.to_datetime(data['date'].unique())))
    n_dates = len(unique_dates)
    if n_dates < 40:
        raise ValueError('Not enough unique dates for walk-forward validation.')

    initial_train_size = max(int(n_dates * min_train_frac), 30)
    remaining = n_dates - initial_train_size
    if remaining < n_folds:
        raise ValueError('Not enough remaining dates to create the requested number of folds.')

    # Split the remaining dates into nearly equal contiguous future blocks.
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
            'train_start_date': pd.Timestamp(train_dates[0]),
            'train_end_date': pd.Timestamp(train_dates[-1]),
            'test_start_date': pd.Timestamp(test_dates[0]),
            'test_end_date': pd.Timestamp(test_dates[-1]),
            'train_dates': train_dates,
            'test_dates': test_dates,
        })
        train_end = test_end
        test_start = test_end
    return folds


# -----------------------------------------------------------------------------
# Modeling utilities
# -----------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols):
    """Shared preprocessing for all model runs."""
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


def evaluate_one_fold(train_df, test_df, features, model, model_name, dataset_name, fold_id):
    """Fit one model on one fold and return metrics + predictions."""
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
        'fold': fold_id,
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
    pred_df['fold'] = fold_id
    pred_df['dataset'] = dataset_name
    pred_df['model'] = model_name
    return pipe, metrics, pred_df


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def main():
    common_df, baseline_features, trend_features, lagged_trend_cols, master_path, lag_path = build_common_feature_table()
    folds = make_walk_forward_folds(common_df, n_folds=4, min_train_frac=0.5)

    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('RandomForest', RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1
        ))
    ]

    results = []
    preds_all = []

    for fold_info in folds:
        fold_id = fold_info['fold']
        train_df = common_df[common_df['date'].isin(fold_info['train_dates'])].copy()
        test_df = common_df[common_df['date'].isin(fold_info['test_dates'])].copy()

        for dataset_name, features in [
            ('baseline_calendar_autoreg_walk_forward', baseline_features),
            ('baseline_plus_trends_walk_forward', trend_features),
        ]:
            for model_name, model in models:
                pipe, metrics, pred_df = evaluate_one_fold(
                    train_df, test_df, features, model, model_name, dataset_name, fold_id
                )
                results.append(metrics)
                preds_all.append(pred_df)

                # Save Random Forest feature importances per fold for trends model.
                if model_name == 'RandomForest':
                    prep = pipe.named_steps['prep']
                    rf = pipe.named_steps['model']
                    feature_names = prep.get_feature_names_out()
                    importances = pd.DataFrame({
                        'feature': feature_names,
                        'importance': rf.feature_importances_,
                        'fold': fold_id,
                        'dataset': dataset_name,
                    }).sort_values('importance', ascending=False)
                    importances.to_csv(
                        MODEL_DIR / f'{dataset_name}_{model_name.lower()}_feature_importance_fold_{fold_id}.csv',
                        index=False,
                    )

    results_df = pd.DataFrame(results).sort_values(['fold', 'dataset', 'RMSE'])
    preds_df = pd.concat(preds_all, ignore_index=True)

    # Aggregate across folds for the key final comparison.
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
        .sort_values(['dataset', 'mean_RMSE'])
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
    by_hotel_df = pd.DataFrame(by_hotel_rows).sort_values(['fold', 'dataset', 'model', 'RMSE'])

    # Save tables.
    results_df.to_csv(MODEL_DIR / 'walk_forward_fold_results.csv', index=False)
    summary_df.to_csv(MODEL_DIR / 'walk_forward_summary.csv', index=False)
    by_hotel_df.to_csv(MODEL_DIR / 'walk_forward_by_hotel.csv', index=False)
    preds_df.to_csv(MODEL_DIR / 'walk_forward_predictions.csv', index=False)

    results_df.to_csv(REPORT_DIR / 'walk_forward_fold_results.csv', index=False)
    summary_df.to_csv(REPORT_DIR / 'walk_forward_summary.csv', index=False)
    by_hotel_df.to_csv(REPORT_DIR / 'walk_forward_by_hotel.csv', index=False)

    # Plot fold-by-fold RMSE comparison.
    pivot_rmse = results_df.pivot_table(index='fold', columns=['dataset', 'model'], values='RMSE')
    plt.figure(figsize=(10, 5))
    for col in pivot_rmse.columns:
        plt.plot(pivot_rmse.index, pivot_rmse[col], marker='o', label=' | '.join(col))
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('Walk-forward RMSE by fold')
    plt.xticks(sorted(results_df['fold'].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'walk_forward_rmse_by_fold.png', dpi=150)
    plt.close()

    # Plot mean RMSE summary as bar chart for easier final presentation.
    plt.figure(figsize=(8, 5))
    labels = summary_df['dataset'] + ' | ' + summary_df['model']
    plt.bar(labels, summary_df['mean_RMSE'])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Mean RMSE across folds')
    plt.title('Walk-forward mean RMSE summary')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'walk_forward_mean_rmse_summary.png', dpi=150)
    plt.close()

    # Compact markdown summary.
    base_best = summary_df[summary_df['dataset'] == 'baseline_calendar_autoreg_walk_forward'].sort_values('mean_RMSE').iloc[0]
    trend_best = summary_df[summary_df['dataset'] == 'baseline_plus_trends_walk_forward'].sort_values('mean_RMSE').iloc[0]
    delta = base_best['mean_RMSE'] - trend_best['mean_RMSE']

    summary_lines = []
    summary_lines.append('# Walk-Forward Validation Summary')
    summary_lines.append('')
    summary_lines.append('## Purpose')
    summary_lines.append('')
    summary_lines.append('This script evaluates baseline vs baseline+trends using expanding-window walk-forward validation on the same aligned sample.')
    summary_lines.append('It tests whether lagged Google Trends improves prediction consistently across multiple future periods, not just one holdout split.')
    summary_lines.append('')
    summary_lines.append('## Inputs used')
    summary_lines.append('')
    summary_lines.append(f'- Master table: `{master_path}`')
    summary_lines.append(f'- Best lag file: `{lag_path}`')
    summary_lines.append(f'- Included lagged Trends: {", ".join(lagged_trend_cols)}')
    summary_lines.append('')
    summary_lines.append('## Fold design')
    summary_lines.append('')
    for fold_info in folds:
        summary_lines.append(
            f"- Fold {fold_info['fold']}: train {fold_info['train_start_date'].date()} to {fold_info['train_end_date'].date()}, "
            f"test {fold_info['test_start_date'].date()} to {fold_info['test_end_date'].date()}"
        )
    summary_lines.append('')
    summary_lines.append('## Fold-level results')
    summary_lines.append('')
    summary_lines.append(results_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## Mean results across folds')
    summary_lines.append('')
    summary_lines.append(summary_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## By-hotel results across folds')
    summary_lines.append('')
    summary_lines.append(by_hotel_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## Interpretation')
    summary_lines.append('')
    summary_lines.append(f"- Best baseline-only mean RMSE across folds: **{base_best['mean_RMSE']:.3f}**.")
    summary_lines.append(f"- Best baseline+trends mean RMSE across folds: **{trend_best['mean_RMSE']:.3f}**.")
    summary_lines.append(f"- Mean RMSE difference (baseline - trends): **{delta:.3f}**. Positive means the trends model performed better on average across folds.")
    summary_lines.append('- This result is stronger than a single holdout because it checks whether the same conclusion survives across multiple future windows.')

    (REPORT_DIR / 'modeling_walk_forward_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
