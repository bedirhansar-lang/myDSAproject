# myDSAproject

Current project structure:
- scripts/eda_first_pass.py -> first-pass EDA, same-day correlations, and lag correlation screening
- scripts/modeling_baseline.py -> baseline occupancy forecasting models with and without lagged Google Trends features
- reports/eda_summary.txt -> first EDA findings
- reports/modeling_summary.txt -> first baseline modeling findings
- reports/model_comparison.csv -> model-level evaluation metrics
- reports/model_comparison_by_hotel.csv -> hotel-level evaluation metrics

Current result:
The lagged Google Trends random forest model outperformed the baseline-only random forest on the held-out time period.

Data note:
Raw Excel files and generated images were kept local and were not committed to the repository in this pass.
