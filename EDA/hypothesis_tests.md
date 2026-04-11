# Hypothesis Tests and Interpretation

## Main idea
This project asks whether Google Trends can help explain or predict hotel occupancy in Antalya/Alanya resort hotels.

## Hypotheses
### H1
Google Trends features have a relationship with hotel occupancy.

### H2
Lagged Google Trends features are more useful than same-day Google Trends features.

### H3
The strength of the relationship depends on country and keyword.

## Tests used in this stage
- Pearson correlation for same-day relationships
- Spearman correlation for rank-based same-day relationships
- Lagged Pearson correlation at 7, 14, 21, and 28 days

## What was found
### H1
Partially supported.
Some Google Trends variables showed weak to moderate same-day relationships with occupancy.

### H2
Supported.
Several lagged Google Trends features showed stronger relationships than same-day versions.

### H3
Supported.
Not all countries and keywords behaved the same. Some Turkiye and Germany related search terms were stronger than many UK terms.

## Caution
These tests are useful for the EDA stage, but they do not prove causality. They only show that some search signals move with occupancy, especially with delay.

## Conclusion in simple terms
Google Trends is not a perfect direct explanation of occupancy, but some lagged search features seem useful enough to carry forward into the machine learning stage.
