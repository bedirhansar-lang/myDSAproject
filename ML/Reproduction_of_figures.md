# ML Figure Reproducibility

The figures shown in `ML/ML_detailed_report.ipynb` are stored in:

- `ML/Figures/`
- `ML/Naive_Benchmark/`

The modeling scripts generate their original plot outputs under `model_outputs/...`.
After running the ML scripts, run the sync script below to copy the generated PNG files into the report figure folders.

## Reproduce ML outputs and report figures

Run these commands from the repository root:

```bash
python scripts/modeling_baseline_commented.py
python scripts/hotel_normalization_robustness_commented.py
python scripts/modeling_fair_comparison_commented.py
python scripts/modeling_walk_forward_commented.py
python scripts/modeling_naive_benchmarks_commented.py
python scripts/sync_ml_report_figures.py
```

## What the sync script does

`scripts/sync_ml_report_figures.py` copies generated plots from `model_outputs/...` into the exact folders used by the ML report.

For example:

```text
model_outputs/walk_forward_validation/walk_forward_rmse_by_fold.png
        -> ML/Figures/walk_forward_rmse_by_fold.png

model_outputs/naive_benchmark_comparison/same_window_rmse_with_naive_benchmarks.png
        -> ML/Naive_Benchmark/same_window_rmse_with_naive_benchmarks.png
```

This means the report figures can be regenerated from the modeling scripts instead of being only manually uploaded images.
