import pandas as pd
import numpy as np
import math
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path("/mnt/data")
OUT = BASE / "model_outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_excel(BASE / "hotel_master_table.xlsx", sheet_name="master_table")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["hotel_name", "date"]).reset_index(drop=True)

df["hotel_code"] = df["hotel_name"].astype("category").cat.codes
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day_of_year"] = df["date"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

for lag in [7, 14, 28]:
    df[f"occupancy_lag_{lag}"] = df.groupby("hotel_name")["occupancy_rate"].shift(lag)

best = pd.read_csv(BASE / "eda_outputs" / "best_lag_correlations.csv")
lagged = []
for _, row in best.head(8).iterrows():
    feat = row["feature"]
    lag = int(row["lag_days"])
    new_col = f"{feat}_lag_{lag}"
    df[new_col] = df[feat].shift(lag)
    lagged.append(new_col)

baseline = [
    "hotel_code", "month", "day_of_week", "week_of_year",
    "doy_sin", "doy_cos", "occupancy_lag_7", "occupancy_lag_14", "occupancy_lag_28"
]

feature_sets = {
    "baseline_only": baseline,
    "baseline_top3": baseline + lagged[:3],
    "baseline_top5": baseline + lagged[:5],
    "baseline_top8": baseline + lagged[:8],
}

rows = []
for name, feats in feature_sets.items():
    d = df.dropna(subset=feats + ["occupancy_rate"]).copy()
    unique_dates = np.array(sorted(d["date"].unique()))
    split1 = int(len(unique_dates) * 0.6)
    split2 = int(len(unique_dates) * 0.8)
    folds = [
        ("fold1", unique_dates[:split1], unique_dates[split1:split2]),
        ("fold2", unique_dates[:split2], unique_dates[split2:]),
    ]

    for fold_name, train_dates, test_dates in folds:
        train = d[d["date"].isin(train_dates)].copy()
        test = d[d["date"].isin(test_dates)].copy()
        X_train = train[feats].values
        y_train = train["occupancy_rate"].values
        X_test = test[feats].values
        y_test = test["occupancy_rate"].values

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows.append({
            "feature_set": name,
            "fold": fold_name,
            "test_start": str(test["date"].min().date()),
            "test_end": str(test["date"].max().date()),
            "train_rows": len(train),
            "test_rows": len(test),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": math.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

folds_df = pd.DataFrame(rows)
summary_df = (
    folds_df.groupby("feature_set", as_index=False)
    .agg(avg_MAE=("MAE", "mean"), avg_RMSE=("RMSE", "mean"), avg_R2=("R2", "mean"))
    .sort_values("avg_RMSE")
)

folds_df.to_csv(OUT / "feature_set_walkforward_folds.csv", index=False)
summary_df.to_csv(OUT / "feature_set_walkforward_summary.csv", index=False)
print(summary_df)
