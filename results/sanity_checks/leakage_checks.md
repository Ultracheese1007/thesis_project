# Day 2 Leakage and Alignment Checks

## 1. Master Calendar
The final panel is constructed on the NASDAQ-100 trading-day calendar.
Reason:
- the dependent variable is daily NASDAQ-100 log return
- rows without a valid market close are excluded from the master panel

## 2. Target Construction
`nasdaq_return` is computed as the daily log return:
- nasdaq_return_t = ln(P_t / P_(t-1))
where t-1 refers to the previous trading day on the master panel.

## 3. Macro Alignment
DGS10 and VIX are merged onto the trading-day panel by exact date.
After merging:
- `dgs10` is forward-filled
- `vix` is forward-filled

Interpretation:
Missing macro observations are treated as unavailable updates rather than nonexistent states.

## 4. Sentiment Alignment
Daily sentiment is merged onto the trading-day panel by exact date.
The retained sentiment variables are:
- sentiment_mean
- sentiment_std
- sentiment_extreme

When a trading day has no aligned sentiment observation:
- sentiment_mean = 0
- sentiment_std = 0
- sentiment_extreme = 0

Interpretation:
This encodes absence of aligned news signal on that trading day, without imposing an arbitrary carry-over rule from non-trading days.

## 5. Lag Policy
The following predictors are lagged by one trading day:
- dgs10_lag1
- vix_lag1
- sentiment_mean_lag1
- sentiment_std_lag1
- sentiment_extreme_lag1

Lagging is performed only after all alignment and filling steps are complete.

## 6. Final Dataset Freeze
`final_daily_dataset.csv` is the unique mother table for the thesis experiment stage.
It contains:
- target
- aligned raw predictors
- lagged predictors used for modeling

## 7. Train/Test Split
Chronological split:
- train: 2020-01-01 to 2023-12-31
- test: 2024-01-01 to 2024-12-31

No random split is used.

## 8. Standardization Policy
Standardization is NOT fit on the full dataset at Day 2.
The frozen rule is:
- fit standardization on training data only
- apply fitted parameters to validation/test data only
- within CV, fit preprocessing only on the fold-specific training block

This is a methodological freeze for Day 3 modeling, not a full-dataset transformation step.
