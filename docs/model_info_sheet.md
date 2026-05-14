# Model Info Sheet — Final RQ-Aligned Daily Thesis Specification

## 1. Thesis Scope

- Thesis framework: DSS Master Thesis / Data Science in Action
- Authoritative frequency: daily
- Authoritative target object: NASDAQ-100 daily return prediction
- Final empirical target: `nasdaq_return`
- The AI technology stock narrative motivates the research context, but the formal empirical target is the NASDAQ-100 daily log return.

## 2. Research Target

- Target column: `nasdaq_return`
- Definition: daily log return of the NASDAQ-100 index
- Forecast timing: predict day t return using predictors available up to day t-1

## 3. Predictor Families

### 3.1 Macro predictors

- `dgs10_lag1`
- `vix_lag1`

### 3.2 Sentiment predictors

- `sentiment_mean_lag1`
- `sentiment_std_lag1`
- `sentiment_extreme_lag1`

These lagged variables are constructed from the raw daily sentiment columns:

- `sentiment_mean`
- `sentiment_std`
- `sentiment_extreme`

## 4. Primary Feature Settings

The final thesis design uses nine RQ-aligned feature specifications.

### 4.1 RQ1 — Macro baseline

- `macro_only`
  - `dgs10_lag1`
  - `vix_lag1`

Purpose:

- evaluate whether macro-financial indicators provide predictive information for NASDAQ-100 daily log returns
- compare linear and nonlinear model performance under the same macro-only input setting

### 4.2 RQ2 — Standalone sentiment representations

- `sentiment_mean_only`
  - `sentiment_mean_lag1`

- `sentiment_std_only`
  - `sentiment_std_lag1`

- `sentiment_extreme_only`
  - `sentiment_extreme_lag1`

Purpose:

- evaluate whether FinBERT-based institutional sentiment representations have standalone predictive information
- distinguish sentiment direction, sentiment dispersion, and extreme sentiment share

### 4.3 RQ3 — Incremental sentiment value beyond macro baseline

- `macro_mean`
  - `dgs10_lag1`
  - `vix_lag1`
  - `sentiment_mean_lag1`

- `macro_std`
  - `dgs10_lag1`
  - `vix_lag1`
  - `sentiment_std_lag1`

- `macro_extreme`
  - `dgs10_lag1`
  - `vix_lag1`
  - `sentiment_extreme_lag1`

- `macro_mean_std`
  - `dgs10_lag1`
  - `vix_lag1`
  - `sentiment_mean_lag1`
  - `sentiment_std_lag1`

- `macro_mean_extreme`
  - `dgs10_lag1`
  - `vix_lag1`
  - `sentiment_mean_lag1`
  - `sentiment_extreme_lag1`

Purpose:

- compare each macro-sentiment specification against `macro_only`
- evaluate whether sentiment features add incremental predictive value beyond the macro baseline
- report incremental value through delta MAE, delta RMSE, and delta R2

## 5. Feature Design Rationale

The final design does not use a single full combined specification as the primary feature setting.

Reason:

1. The thesis aims to isolate different sentiment representations.
2. `sentiment_mean_lag1`, `sentiment_std_lag1`, and `sentiment_extreme_lag1` capture different concepts:
   - sentiment direction
   - sentiment dispersion
   - extreme sentiment share
3. `sentiment_std_lag1` and `sentiment_extreme_lag1` are conceptually related and empirically highly correlated.
4. Separating these variables improves interpretability, especially for OLS.
5. Limited combined sentiment settings, namely `macro_mean_std` and `macro_mean_extreme`, are retained for RQ3.

## 6. Lag Policy

All predictors are lagged by one trading day.

Rationale:

1. preserve temporal ordering
2. avoid look-ahead bias
3. ensure consistent timing across macro and sentiment predictors

## 7. Train/Test Split

- Train period: 2020-01-03 to 2023-12-29
- Number of training observations: 1005
- Test period: 2024-01-02 to 2024-12-31
- Number of test observations: 252
- Split type: chronological only
- Random split: not allowed

## 8. Cross-Validation

- Type: expanding-window time-series cross-validation
- Scope: training set only
- Number of folds: 3
- Shuffle: no
- Future information leakage: not allowed

Fold logic:

- Fold 1: earliest training block -> next chronological validation block
- Fold 2: expanded training block -> next chronological validation block
- Fold 3: further expanded training block -> final chronological validation block before 2024

## 9. Models

Primary models:

- OLS
- Random Forest
- XGBoost

Primary comparison matrix:

- OLS × 9 feature specifications
- Random Forest × 9 feature specifications
- XGBoost × 9 feature specifications

Total primary model-setting combinations:

- 3 model families × 9 feature specifications = 27 combinations

## 10. Evaluation Metrics

Primary metrics:

- MAE
- RMSE
- R2

Incremental value metrics for RQ3:

- delta_MAE = MAE(candidate specification) - MAE(macro_only)
- delta_RMSE = RMSE(candidate specification) - RMSE(macro_only)
- delta_R2 = R2(candidate specification) - R2(macro_only)

Interpretation:

- Negative delta_MAE indicates lower prediction error than the macro-only baseline.
- Negative delta_RMSE indicates lower prediction error than the macro-only baseline.
- Positive delta_R2 indicates higher explanatory performance than the macro-only baseline.

RQ3 candidate specifications:

- `macro_mean`
- `macro_std`
- `macro_extreme`
- `macro_mean_std`
- `macro_mean_extreme`

Reference specification:

- `macro_only`

## 11. Model Selection Rule

For nonlinear models, hyperparameter tuning is conducted inside the training period only.

Primary selection metric:

- lowest mean cross-validation RMSE

Tie-breaking rule:

1. higher mean cross-validation R2
2. lower grid_order, corresponding to the simpler-first grid ordering

Mean cross-validation MAE is reported as an evaluation metric, but it is not used as the formal hyperparameter selection criterion.

The 2024 hold-out test set is not used for hyperparameter tuning.

## 12. Error Analysis Scope

The thesis will include:

- predicted vs actual
- residuals over time
- rolling-window error behaviour
- high-VIX regime error
- extreme-sentiment regime error

## 13. SHAP Interpretation Scope

SHAP is used for tuned nonlinear models to examine model-based predictive contribution.

Important interpretation rule:

- SHAP values do not imply causal effects.
- SHAP shows how a trained model uses features for prediction.
- SHAP should be interpreted together with out-of-sample metrics and error analysis.

## 14. Benchmark Status: Lagged Target

The proposal states that the lagged target term may be constructed as a benchmark predictor.

Final decision:

- acknowledge this benchmark idea
- do not include it in the primary nine feature specifications
- keep the primary design focused on macro predictors, sentiment representations, and macro-sentiment combinations

## 15. Leakage Control Rules

1. all predictors must be lagged before modeling
2. no future information may enter any training fold
3. standardization must be fitted only on the relevant training portion
4. validation and test data must be transformed using previously fitted parameters only
5. no random shuffling is allowed in splitting or CV
6. the 2024 hold-out test set is not used during model selection or hyperparameter tuning

## 16. What Is Finalized

The following elements are finalized for the thesis experiment:

- target definition
- predictor definitions
- nine RQ-aligned feature specifications
- lag policy
- train/test split
- expanding-window CV strategy
- model comparison matrix
- evaluation metrics
- RQ3 delta metric definitions
- nonlinear model selection rule
- error analysis scope
- SHAP interpretation scope
- benchmark status of lagged target
