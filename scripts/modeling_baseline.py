
import pandas as pd
import numpy as np
import math
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

BASE_DIR = Path("/mnt/data")
OUT_DIR = BASE_DIR / "model_outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_excel(BASE_DIR / "hotel_master_table.xlsx", sheet_name="master_table")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["hotel_name", "date"]).reset_index(drop=True)

# Calendar features
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day_of_year"] = df["date"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# Occupancy lags by hotel
for lag in [7, 14, 28]:
    df[f"occupancy_lag_{lag}"] = df.groupby("hotel_name")["occupancy_rate"].shift(lag)

# Best lagged Google Trends features from EDA
best = pd.read_csv(BASE_DIR / "eda_outputs" / "best_lag_correlations.csv")
top_best = best.head(8).copy()
lagged_trend_cols = []
for _, row in top_best.iterrows():
    feat = row["feature"]
    lag = int(row["lag_days"])
    new_col = f"{feat}_lag_{lag}"
    df[new_col] = df[feat].shift(lag)
    lagged_trend_cols.append(new_col)
lagged_trend_cols = list(dict.fromkeys(lagged_trend_cols))

model_df = df.dropna(subset=["occupancy_rate"]).copy()

baseline_features = [
    "hotel_name", "month", "day_of_week", "week_of_year", "doy_sin", "doy_cos",
    "occupancy_lag_7", "occupancy_lag_14", "occupancy_lag_28"
]
trend_features = baseline_features + lagged_trend_cols

baseline_df = model_df.dropna(subset=baseline_features).copy()
trend_df = model_df.dropna(subset=trend_features).copy()

def time_split(data, test_frac=0.2):
    unique_dates = np.array(sorted(pd.to_datetime(data["date"].unique())))
    split_idx = int(len(unique_dates) * (1 - test_frac))
    split_date = pd.Timestamp(unique_dates[split_idx])
    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()
    return train, test, split_date

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def evaluate_model(train_df, test_df, features, model, model_name, dataset_name):
    X_train = train_df[features]
    y_train = train_df["occupancy_rate"]
    X_test = test_df[features]
    y_test = test_df["occupancy_rate"]

    cat_cols = [c for c in features if X_train[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    pipe = Pipeline([
        ("prep", build_preprocessor(num_cols, cat_cols)),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "dataset": dataset_name,
        "model": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
        "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": math.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

    pred_df = test_df[["date", "hotel_name", "occupancy_rate"]].copy()
    pred_df["prediction"] = preds
    pred_df["model"] = model_name
    pred_df["dataset"] = dataset_name
    return pipe, metrics, pred_df

base_train, base_test, _ = time_split(baseline_df)
trend_train, trend_test, _ = time_split(trend_df)

results = []
preds_all = []

models = [
    ("Ridge", Ridge(alpha=1.0)),
    ("RandomForest", RandomForestRegressor(
        n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1
    ))
]

for dataset_name, train_df, test_df, features in [
    ("baseline_calendar_autoreg", base_train, base_test, baseline_features),
    ("baseline_plus_trends", trend_train, trend_test, trend_features),
]:
    for model_name, model in models:
        pipe, metrics, pred_df = evaluate_model(train_df, test_df, features, model, model_name, dataset_name)
        results.append(metrics)
        preds_all.append(pred_df)

        if model_name == "RandomForest":
            prep = pipe.named_steps["prep"]
            rf = pipe.named_steps["model"]
            feature_names = prep.get_feature_names_out()
            importances = pd.DataFrame({
                "feature": feature_names,
                "importance": rf.feature_importances_
            }).sort_values("importance", ascending=False)
            importances.to_csv(OUT_DIR / f"{dataset_name}_{model_name.lower()}_feature_importance.csv", index=False)

results_df = pd.DataFrame(results).sort_values(["dataset", "RMSE"])
preds_df = pd.concat(preds_all, ignore_index=True)

by_hotel_rows = []
for (dataset, model_name), g in preds_df.groupby(["dataset", "model"]):
    for hotel, hg in g.groupby("hotel_name"):
        by_hotel_rows.append({
            "dataset": dataset,
            "model": model_name,
            "hotel_name": hotel,
            "test_rows": len(hg),
            "MAE": mean_absolute_error(hg["occupancy_rate"], hg["prediction"]),
            "RMSE": math.sqrt(mean_squared_error(hg["occupancy_rate"], hg["prediction"])),
            "R2": r2_score(hg["occupancy_rate"], hg["prediction"]),
        })
by_hotel_df = pd.DataFrame(by_hotel_rows).sort_values(["dataset", "model", "RMSE"])

selected_cols = ["date", "hotel_name", "occupancy_rate"] + baseline_features[1:] + lagged_trend_cols
trend_df[selected_cols].to_excel(OUT_DIR / "modeling_dataset_with_lags.xlsx", index=False)

results_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
by_hotel_df.to_csv(OUT_DIR / "model_comparison_by_hotel.csv", index=False)
preds_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

best_row = results_df[results_df["dataset"] == "baseline_plus_trends"].sort_values("RMSE").iloc[0]
best_preds = preds_df[
    (preds_df["dataset"] == best_row["dataset"]) &
    (preds_df["model"] == best_row["model"])
].copy()

for hotel, g in best_preds.groupby("hotel_name"):
    g = g.sort_values("date")
    plt.figure(figsize=(12, 4))
    plt.plot(g["date"], g["occupancy_rate"], label="Actual")
    plt.plot(g["date"], g["prediction"], label="Predicted")
    plt.title(f"{hotel}: Actual vs Predicted Occupancy ({best_row['model']}, trends model)")
    plt.xlabel("Date")
    plt.ylabel("Occupancy Rate")
    plt.legend()
    plt.tight_layout()
    fname = hotel.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(OUT_DIR / f"actual_vs_pred_{fname}.png", dpi=150)
    plt.close()

summary_lines = []
summary_lines.append("# Baseline Modeling Summary")
summary_lines.append("")
summary_lines.append("## Setup")
summary_lines.append("")
summary_lines.append("- Target: `occupancy_rate`")
summary_lines.append("- Time-aware split: final 20% of unique dates used as test set.")
summary_lines.append("- Baseline features: hotel identity, calendar seasonality, and occupancy lags (7/14/28 days).")
summary_lines.append("- Trends model: baseline features plus top lagged Google Trends features selected from the first EDA pass.")
summary_lines.append(f"- Top lagged Trends included: {', '.join(lagged_trend_cols)}")
summary_lines.append("")
summary_lines.append("## Model comparison")
summary_lines.append("")
summary_lines.append(results_df.to_markdown(index=False))
summary_lines.append("")
summary_lines.append("## By-hotel performance")
summary_lines.append("")
summary_lines.append(by_hotel_df.to_markdown(index=False))
summary_lines.append("")
summary_lines.append("## Interpretation")
summary_lines.append("")
best_overall = results_df.sort_values("RMSE").iloc[0]
base_best = results_df[results_df["dataset"] == "baseline_calendar_autoreg"].sort_values("RMSE").iloc[0]
trend_best = results_df[results_df["dataset"] == "baseline_plus_trends"].sort_values("RMSE").iloc[0]
delta = base_best["RMSE"] - trend_best["RMSE"]
summary_lines.append(f"- Best overall test RMSE: **{best_overall['RMSE']:.3f}** from **{best_overall['dataset']} / {best_overall['model']}**.")
summary_lines.append(f"- Best baseline-only RMSE: **{base_best['RMSE']:.3f}**.")
summary_lines.append(f"- Best trends-augmented RMSE: **{trend_best['RMSE']:.3f}**.")
summary_lines.append(f"- RMSE difference (baseline - trends): **{delta:.3f}**. Positive means Trends helped.")
summary_lines.append("- This is still a first-pass result. It tests whether lagged Google Trends adds incremental signal beyond hotel seasonality and recent occupancy.")

(OUT_DIR / "modeling_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
