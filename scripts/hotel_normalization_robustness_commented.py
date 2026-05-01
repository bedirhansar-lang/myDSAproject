import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / 'reports'
MODEL_DIR = REPO_ROOT / 'model_outputs'
ROBUST_DIR = MODEL_DIR / 'hotel_normalization_robustness'
REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
ROBUST_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers to discover files even if the repo layout changes slightly.
# -----------------------------------------------------------------------------
def find_first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def find_master_table_path():
    explicit_candidates = [
        REPO_ROOT / 'hotel_master_table.xlsx',
        REPO_ROOT / 'data' / 'hotel_master_table.xlsx',
        REPO_ROOT / 'outputs' / 'hotel_master_table.xlsx',
        REPO_ROOT / 'hotel_master_table.csv',
        REPO_ROOT / 'data' / 'hotel_master_table.csv',
        REPO_ROOT / 'outputs' / 'hotel_master_table.csv',
    ]
    found = find_first_existing(explicit_candidates)
    if found is not None:
        return found

    patterns = [
        '**/hotel_master_table.xlsx',
        '**/hotel_master_table.csv',
        '**/*master_table*.xlsx',
        '**/*master_table*.csv',
    ]
    matches = []
    for pattern in patterns:
        matches.extend([p for p in REPO_ROOT.glob(pattern) if '.git' not in p.parts])
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError('Could not find a hotel master table file in the repository.')
    return matches[0]


def find_best_lag_path():
    explicit_candidates = [
        REPO_ROOT / 'eda_outputs' / 'best_lag_correlations.csv',
        REPO_ROOT / 'outputs' / 'best_lag_correlations.csv',
        REPO_ROOT / 'reports' / 'best_lag_correlations.csv',
        REPO_ROOT / 'best_lag_correlations.csv',
    ]
    found = find_first_existing(explicit_candidates)
    if found is not None:
        return found

    matches = sorted(
        set(
            p for p in REPO_ROOT.glob('**/best_lag_correlations.csv')
            if '.git' not in p.parts
        )
    )
    if not matches:
        raise FileNotFoundError('Could not find best_lag_correlations.csv in the repository.')
    return matches[0]


# -----------------------------------------------------------------------------
# Loading and feature construction.
# -----------------------------------------------------------------------------
def load_master_table():
    master_path = find_master_table_path()
    if master_path.suffix.lower() == '.xlsx':
        try:
            df = pd.read_excel(master_path, sheet_name='master_table')
        except Exception:
            df = pd.read_excel(master_path)
    else:
        df = pd.read_csv(master_path)

    if 'date' not in df.columns or 'hotel_name' not in df.columns or 'occupancy_rate' not in df.columns:
        raise ValueError('Master table must contain date, hotel_name, and occupancy_rate columns.')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['hotel_name', 'date']).reset_index(drop=True)
    return df, master_path


def add_calendar_and_lags(df):
    out = df.copy()
    out['month'] = out['date'].dt.month
    out['day_of_week'] = out['date'].dt.dayofweek
    out['week_of_year'] = out['date'].dt.isocalendar().week.astype(int)
    out['day_of_year'] = out['date'].dt.dayofyear
    out['doy_sin'] = np.sin(2 * np.pi * out['day_of_year'] / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * out['day_of_year'] / 365.25)

    for lag in [7, 14, 28]:
        out[f'occupancy_lag_{lag}'] = out.groupby('hotel_name')['occupancy_rate'].shift(lag)
    return out


def add_selected_trend_lags(df):
    out = df.copy()
    lag_path = find_best_lag_path()
    best = pd.read_csv(lag_path)
    top_best = best.head(8).copy()

    lagged_trend_cols = []
    for _, row in top_best.iterrows():
        feat = row['feature']
        lag = int(row['lag_days'])
        if feat not in out.columns:
            continue
        new_col = f'{feat}_lag_{lag}'
        out[new_col] = out[feat].shift(lag)
        lagged_trend_cols.append(new_col)

    lagged_trend_cols = list(dict.fromkeys(lagged_trend_cols))
    return out, lagged_trend_cols, lag_path


# -----------------------------------------------------------------------------
# EDA robustness using hotel-wise normalized occupancy.
# -----------------------------------------------------------------------------
def add_hotelwise_zscore(df):
    out = df.copy()
    stats = out.groupby('hotel_name')['occupancy_rate'].agg(['mean', 'std']).rename(
        columns={'mean': 'hotel_occ_mean', 'std': 'hotel_occ_std'}
    )
    out = out.merge(stats, left_on='hotel_name', right_index=True, how='left')
    out['occupancy_rate_hotel_z'] = (out['occupancy_rate'] - out['hotel_occ_mean']) / out['hotel_occ_std']
    return out


def safe_corr(x, y):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 10:
        return np.nan
    if valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return np.nan
    return valid.iloc[:, 0].corr(valid.iloc[:, 1])


# -----------------------------------------------------------------------------
# Time-aware split and hotel-wise normalized target for ML robustness.
# -----------------------------------------------------------------------------
def time_split(data, test_frac=0.2):
    unique_dates = np.array(sorted(pd.to_datetime(data['date'].unique())))
    split_idx = int(len(unique_dates) * (1 - test_frac))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])
    train = data[data['date'] < split_date].copy()
    test = data[data['date'] >= split_date].copy()
    return train, test, split_date


def hotel_train_stats(train_df):
    stats = train_df.groupby('hotel_name')['occupancy_rate'].agg(['mean', 'std']).rename(
        columns={'mean': 'train_hotel_mean', 'std': 'train_hotel_std'}
    )
    return stats


def apply_train_based_normalization(df, stats):
    out = df.merge(stats, left_on='hotel_name', right_index=True, how='left').copy()
    out['target_hotel_z'] = (out['occupancy_rate'] - out['train_hotel_mean']) / out['train_hotel_std']
    return out


def build_preprocessor(num_cols, cat_cols):
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


def inverse_hotelwise_z(pred_z, hotel_names, stats):
    stats_map = stats.to_dict('index')
    pred_raw = []
    for pred, hotel in zip(pred_z, hotel_names):
        mu = stats_map[hotel]['train_hotel_mean']
        sd = stats_map[hotel]['train_hotel_std']
        pred_raw.append(mu + pred * sd)
    return np.array(pred_raw)


def evaluate_model(train_df, test_df, features, model, model_name, dataset_name):
    X_train = train_df[features]
    y_train = train_df['target_hotel_z']
    X_test = test_df[features]
    y_test_z = test_df['target_hotel_z']
    y_test_raw = test_df['occupancy_rate']

    cat_cols = [c for c in features if X_train[c].dtype == 'object']
    num_cols = [c for c in features if c not in cat_cols]

    pipe = Pipeline([
        ('prep', build_preprocessor(num_cols, cat_cols)),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    pred_z = pipe.predict(X_test)
    pred_raw = inverse_hotelwise_z(pred_z, test_df['hotel_name'], test_df[['hotel_name', 'train_hotel_mean', 'train_hotel_std']].drop_duplicates().set_index('hotel_name'))

    metrics = {
        'dataset': dataset_name,
        'model': model_name,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'test_start_date': test_df['date'].min().strftime('%Y-%m-%d'),
        'test_end_date': test_df['date'].max().strftime('%Y-%m-%d'),
        'MAE_hotel_z': mean_absolute_error(y_test_z, pred_z),
        'RMSE_hotel_z': math.sqrt(mean_squared_error(y_test_z, pred_z)),
        'R2_hotel_z': r2_score(y_test_z, pred_z),
        'MAE_raw_backtransformed': mean_absolute_error(y_test_raw, pred_raw),
        'RMSE_raw_backtransformed': math.sqrt(mean_squared_error(y_test_raw, pred_raw)),
        'R2_raw_backtransformed': r2_score(y_test_raw, pred_raw),
    }

    pred_df = test_df[['date', 'hotel_name', 'occupancy_rate', 'target_hotel_z']].copy()
    pred_df['prediction_hotel_z'] = pred_z
    pred_df['prediction_raw_backtransformed'] = pred_raw
    pred_df['model'] = model_name
    pred_df['dataset'] = dataset_name
    return pipe, metrics, pred_df


def main():
    df, master_path = load_master_table()
    df = add_calendar_and_lags(df)
    df, lagged_trend_cols, lag_path = add_selected_trend_lags(df)
    df = add_hotelwise_zscore(df)

    # --------------------------
    # EDA robustness analysis
    # --------------------------
    trend_cols = [c for c in df.columns if c.startswith('trends_') and not c.endswith(tuple(['_lag_7', '_lag_14', '_lag_21', '_lag_28']))]
    same_day_rows = []
    for col in trend_cols:
        r_raw = safe_corr(df[col], df['occupancy_rate'])
        r_norm = safe_corr(df[col], df['occupancy_rate_hotel_z'])
        same_day_rows.append({
            'feature': col,
            'pearson_raw': r_raw,
            'pearson_hotel_z': r_norm,
            'abs_delta': abs(r_norm) - abs(r_raw) if pd.notna(r_raw) and pd.notna(r_norm) else np.nan,
        })
    same_day_df = pd.DataFrame(same_day_rows).sort_values('pearson_hotel_z', ascending=False)

    lag_rows = []
    for col in trend_cols:
        for lag in [7, 14, 21, 28]:
            shifted = df[col].shift(lag)
            r_raw = safe_corr(shifted, df['occupancy_rate'])
            r_norm = safe_corr(shifted, df['occupancy_rate_hotel_z'])
            lag_rows.append({
                'feature': col,
                'lag_days': lag,
                'pearson_raw': r_raw,
                'pearson_hotel_z': r_norm,
                'abs_delta': abs(r_norm) - abs(r_raw) if pd.notna(r_raw) and pd.notna(r_norm) else np.nan,
            })
    lag_df = pd.DataFrame(lag_rows)
    best_lag_norm_df = lag_df.sort_values('pearson_hotel_z', ascending=False)

    same_day_df.to_csv(ROBUST_DIR / 'same_day_correlations_hotel_normalized.csv', index=False)
    lag_df.to_csv(ROBUST_DIR / 'lag_correlations_hotel_normalized.csv', index=False)
    best_lag_norm_df.head(20).to_csv(ROBUST_DIR / 'best_lag_correlations_hotel_normalized.csv', index=False)

    # --------------------------
    # ML robustness analysis
    # --------------------------
    model_df = df.dropna(subset=['occupancy_rate']).copy()
    baseline_features = [
        'hotel_name', 'month', 'day_of_week', 'week_of_year', 'doy_sin', 'doy_cos',
        'occupancy_lag_7', 'occupancy_lag_14', 'occupancy_lag_28'
    ]
    trend_features = baseline_features + lagged_trend_cols

    baseline_df = model_df.dropna(subset=baseline_features).copy()
    trend_df = model_df.dropna(subset=trend_features).copy()

    base_train, base_test, split_date_base = time_split(baseline_df)
    trend_train, trend_test, split_date_trend = time_split(trend_df)

    base_stats = hotel_train_stats(base_train)
    trend_stats = hotel_train_stats(trend_train)

    base_train = apply_train_based_normalization(base_train, base_stats).dropna(subset=['target_hotel_z'])
    base_test = apply_train_based_normalization(base_test, base_stats).dropna(subset=['target_hotel_z'])
    trend_train = apply_train_based_normalization(trend_train, trend_stats).dropna(subset=['target_hotel_z'])
    trend_test = apply_train_based_normalization(trend_test, trend_stats).dropna(subset=['target_hotel_z'])

    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('RandomForest', RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1
        ))
    ]

    results = []
    preds_all = []
    for dataset_name, train_df, test_df, features in [
        ('baseline_calendar_autoreg_hotel_z_target', base_train, base_test, baseline_features),
        ('baseline_plus_trends_hotel_z_target', trend_train, trend_test, trend_features),
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
                importances.to_csv(
                    ROBUST_DIR / f'{dataset_name}_{model_name.lower()}_feature_importance.csv',
                    index=False
                )

    results_df = pd.DataFrame(results).sort_values(['dataset', 'RMSE_raw_backtransformed'])
    preds_df = pd.concat(preds_all, ignore_index=True)

    results_df.to_csv(ROBUST_DIR / 'model_comparison_hotel_normalized_target.csv', index=False)
    preds_df.to_csv(ROBUST_DIR / 'test_predictions_hotel_normalized_target.csv', index=False)

    by_hotel_rows = []
    for (dataset, model_name), g in preds_df.groupby(['dataset', 'model']):
        for hotel, hg in g.groupby('hotel_name'):
            by_hotel_rows.append({
                'dataset': dataset,
                'model': model_name,
                'hotel_name': hotel,
                'test_rows': len(hg),
                'MAE_hotel_z': mean_absolute_error(hg['target_hotel_z'], hg['prediction_hotel_z']),
                'RMSE_hotel_z': math.sqrt(mean_squared_error(hg['target_hotel_z'], hg['prediction_hotel_z'])),
                'R2_hotel_z': r2_score(hg['target_hotel_z'], hg['prediction_hotel_z']),
                'MAE_raw_backtransformed': mean_absolute_error(hg['occupancy_rate'], hg['prediction_raw_backtransformed']),
                'RMSE_raw_backtransformed': math.sqrt(mean_squared_error(hg['occupancy_rate'], hg['prediction_raw_backtransformed'])),
                'R2_raw_backtransformed': r2_score(hg['occupancy_rate'], hg['prediction_raw_backtransformed']),
            })
    by_hotel_df = pd.DataFrame(by_hotel_rows).sort_values(['dataset', 'model', 'RMSE_raw_backtransformed'])
    by_hotel_df.to_csv(ROBUST_DIR / 'model_comparison_by_hotel_hotel_normalized_target.csv', index=False)

    best_row = results_df[results_df['dataset'] == 'baseline_plus_trends_hotel_z_target'].sort_values('RMSE_raw_backtransformed').iloc[0]
    best_preds = preds_df[
        (preds_df['dataset'] == best_row['dataset']) &
        (preds_df['model'] == best_row['model'])
    ].copy()

    for hotel, g in best_preds.groupby('hotel_name'):
        g = g.sort_values('date')
        plt.figure(figsize=(12, 4))
        plt.plot(g['date'], g['occupancy_rate'], label='Actual raw occupancy')
        plt.plot(g['date'], g['prediction_raw_backtransformed'], label='Predicted raw occupancy')
        plt.title(f'{hotel}: Actual vs Predicted Occupancy (hotel-normalized target robustness)')
        plt.xlabel('Date')
        plt.ylabel('Occupancy Rate')
        plt.legend()
        plt.tight_layout()
        fname = hotel.lower().replace(' ', '_').replace('/', '_')
        plt.savefig(ROBUST_DIR / f'actual_vs_pred_hotel_z_{fname}.png', dpi=150)
        plt.close()

    # --------------------------
    # Compact text summary for the report folder.
    # --------------------------
    summary_lines = []
    summary_lines.append('# Hotel-wise Normalization Robustness Summary')
    summary_lines.append('')
    summary_lines.append('## Purpose')
    summary_lines.append('')
    summary_lines.append('This robustness pass checks whether joint conclusions remain similar after normalizing occupancy within each hotel.')
    summary_lines.append('The goal is to reduce the influence of between-hotel level differences when comparing pooled relationships and pooled model behavior.')
    summary_lines.append('')
    summary_lines.append('## Inputs discovered automatically')
    summary_lines.append('')
    summary_lines.append(f'- Master table: `{master_path}`')
    summary_lines.append(f'- Best lag file: `{lag_path}`')
    summary_lines.append('')
    summary_lines.append('## EDA robustness setup')
    summary_lines.append('')
    summary_lines.append('- Hotel-wise normalized occupancy variable: `occupancy_rate_hotel_z = (occupancy_rate - hotel_mean) / hotel_std`')
    summary_lines.append('- Same-day and lagged Pearson correlations are recomputed against the normalized occupancy variable.')
    summary_lines.append('- Compare these results against the original raw-occupancy correlations before making pooled cross-hotel claims.')
    summary_lines.append('')
    summary_lines.append('### Top same-day correlations with hotel-normalized occupancy')
    summary_lines.append('')
    summary_lines.append(same_day_df.head(10).to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('### Top lagged correlations with hotel-normalized occupancy')
    summary_lines.append('')
    summary_lines.append(best_lag_norm_df.head(10).to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## ML robustness setup')
    summary_lines.append('')
    summary_lines.append('- Same baseline and baseline+trends feature sets are reused.')
    summary_lines.append('- The target is normalized within hotel using training-period hotel mean and standard deviation only, to avoid leakage.')
    summary_lines.append('- Predictions are also back-transformed into raw occupancy units so business-scale RMSE can still be compared.')
    summary_lines.append(f'- Baseline split date: `{split_date_base.date()}`')
    summary_lines.append(f'- Trends split date: `{split_date_trend.date()}`')
    summary_lines.append('')
    summary_lines.append('### Model comparison on hotel-normalized target')
    summary_lines.append('')
    summary_lines.append(results_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('### By-hotel model comparison on hotel-normalized target')
    summary_lines.append('')
    summary_lines.append(by_hotel_df.to_markdown(index=False))
    summary_lines.append('')
    summary_lines.append('## Interpretation guide')
    summary_lines.append('')
    summary_lines.append('- If the same Google Trends features remain strong after hotel-wise normalization, then the pooled signal is less likely to be driven only by level differences between hotels.')
    summary_lines.append('- If the baseline+trends model still outperforms the baseline model under the normalized-target setup, then the incremental value of Trends is more robust.')
    summary_lines.append('- If performance gains disappear, then earlier pooled gains may have been partly driven by between-hotel differences rather than within-hotel variation.')

    (REPORT_DIR / 'hotel_normalization_robustness_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
