# EDA

This folder contains the work prepared for the 14 April checkpoint: data collection, exploratory data analysis, and hypothesis testing.

## Scope of this stage
- Collect hotel occupancy data for two hotels
- Collect Google Trends data for multiple countries and keywords
- Clean and merge the datasets into one master table
- Explore occupancy patterns and Google Trends behavior
- Test whether Google Trends has a relationship with occupancy, especially with time lags

## Files in this folder
- `checkpoint_summary.md`: short checkpoint-oriented writeup
- `hypothesis_tests.md`: hypotheses and results in simple terms
- `Visualizations/`: notes about the visual outputs created during EDA

## Main dataset used
The main combined dataset was created as `hotel_master_table.xlsx` outside the repository runtime and used for EDA and modeling.

## Main EDA conclusion
Same-day Google Trends relationships are mostly weak to moderate, but several lagged Google Trends features show stronger relationships with occupancy. This supports the idea that search behavior may be useful as an early signal rather than a same-day signal.
