This folder is reserved for the visual outputs of the EDA stage.

Current visuals created locally during the first EDA pass include:
1. occupancy_over_time.png
Shows daily occupancy rate over time for both hotels. Purpose: to reveal seasonality, large drops, and similarity or differences between the hotels.

2. monthly_occupancy.png
Shows monthly average occupancy for both hotels. Purpose: to make the seasonal pattern easier to read than the daily plot.

3. top_same_day_correlations.png
Shows the Google Trends features with the strongest same-day Pearson correlation with occupancy. Purpose: to identify whether any search features are directly related to same-day occupancy.

4. top_lagged_correlations.png
Shows the strongest lagged Google Trends features and their lag values. Purpose: to test whether search interest becomes more useful when shifted earlier in time.

5. best_lag_overlay.png
Overlays normalized occupancy with the strongest lagged Google Trends signal. Purpose: to visually compare whether the strongest delayed Trends feature moves in a similar pattern to occupancy.

These visuals support the work on data collection, EDA, and hypothesis testing.
