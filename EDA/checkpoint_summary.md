# 14 April Checkpoint Summary

## Goal
The goal of this stage is to complete data collection, exploratory data analysis, and hypothesis testing on the hotel occupancy dataset enriched with Google Trends data.

## Data collected
### Hotel data
- Side Mare Hotel daily occupancy data
- Azura Deluxe daily occupancy data
- Target variable used in this stage: `occupancy_rate`

### Enrichment data
Google Trends data was collected for four countries with tourism-related keywords.
Countries included:
- Germany
- Netherlands
- United Kingdom
- Turkiye

## Data preparation done
- Daily occupancy files were cleaned and standardized
- Google Trends files were cleaned and standardized
- All files were merged into one master table by date
- Each row in the master table represents one hotel on one date
- Duplicate date-hotel rows were checked and none were found
- Missing values were checked

## EDA done
- Occupancy was summarized by hotel
- Occupancy was plotted over time
- Monthly occupancy patterns were examined
- Same-day correlations between Google Trends and occupancy were computed
- Lagged correlations at 7, 14, 21, and 28 days were computed

## Main findings
- Occupancy has strong seasonal structure
- Same-day Google Trends relationships are mostly weak to moderate
- Some lagged Google Trends features are stronger than same-day features
- The strongest current signal appeared for `trends_turkiye_side_otel` with a 28-day lag
- This suggests Google Trends may work better as an early signal than a same-day signal

## Status relative to course timeline
This folder covers the required 14 April stage:
- data collection
- EDA
- hypothesis testing

Machine learning work was started separately after this stage and should be treated as the next milestone, not as the main content of this checkpoint.
