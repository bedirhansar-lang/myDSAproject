import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load the merged hotel + trends dataset.
df = pd.read_excel('/mnt/data/hotel_master_table.xlsx', sheet_name='master_table')

# Identify all Google Trends feature columns.
trend_cols = [c for c in df.columns if c.startswith('trends_')]

# Quick dataset checks.
print(df.shape)
print(df.dtypes)
print(df.isna().sum().sort_values(ascending=False))

# Compare occupancy summary statistics by hotel.
print(df.groupby('hotel_name')['occupancy_rate'].agg(['count', 'mean', 'median', 'std', 'min', 'max']))

# Same-day correlation screen.
rows = []
for c in trend_cols:
    sub = df[['occupancy_rate', c]].dropna()
    if len(sub) > 10 and sub[c].nunique() > 1:
        pear = pearsonr(sub['occupancy_rate'], sub[c])
        spear = spearmanr(sub['occupancy_rate'], sub[c], nan_policy='omit')
        rows.append({
            'feature': c,
            'n': len(sub),
            'pearson_r': pear.statistic,
            'pearson_p': pear.pvalue,
            'spearman_rho': spear.statistic,
            'spearman_p': spear.pvalue
        })

same_corr = pd.DataFrame(rows).sort_values('pearson_r', ascending=False)
print(same_corr.head(10))

# Build one unique daily trends table for lag testing.
trend_df = df[['date'] + trend_cols].drop_duplicates('date').sort_values('date').reset_index(drop=True)
lags = [7, 14, 21, 28]
lag_rows = []
for lag in lags:
    lagged = trend_df.copy()
    lagged[trend_cols] = lagged[trend_cols].shift(lag)
    merged = df[['date', 'hotel_name', 'occupancy_rate']].merge(lagged, on='date', how='left')
    for c in trend_cols:
        sub = merged[['occupancy_rate', c]].dropna()
        if len(sub) > 10 and sub[c].nunique() > 1:
            r, p = pearsonr(sub['occupancy_rate'], sub[c])
            lag_rows.append({'lag_days': lag, 'feature': c, 'n': len(sub), 'pearson_r': r, 'pearson_p': p})

# Keep the best lag for each feature.
lag_corr = pd.DataFrame(lag_rows)
best_by_feature = lag_corr.loc[lag_corr.groupby('feature')['pearson_r'].idxmax()].sort_values('pearson_r', ascending=False)
print(best_by_feature.head(10))
