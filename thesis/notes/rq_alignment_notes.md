# RQ Alignment Notes — Final RQ-Aligned Daily Thesis Specification

## 1. Authoritative Thesis Object

The authoritative thesis object is:

- daily NASDAQ-100 return prediction

The thesis is not formally specified as:

- monthly return prediction
- mixed monthly/daily wording
- individual AI-tech stock panel prediction

The AI technology market narrative motivates the research context, but the empirical target is the NASDAQ-100 daily log return.

## 2. RQ1 Alignment

### RQ1

To what extent can macroeconomic indicators, represented by DGS10 and VIX, predict NASDAQ-100 daily log returns, and do predictive performances differ between linear and nonlinear models?

### Empirical answer path

Feature setting:

- `macro_only`

Variables:

- `dgs10_lag1`
- `vix_lag1`

Model comparison:

- OLS
- Random Forest
- XGBoost

Evaluation:

- expanding-window CV inside the training period
- 2024 out-of-sample hold-out test
- MAE, RMSE, and R2

Interpretation:

- answers whether macro-financial variables provide a meaningful but limited predictive baseline

## 3. RQ2 Alignment

### RQ2

To what extent do FinBERT-based institutional sentiment representations contain standalone predictive information for NASDAQ-100 daily log returns?

### Empirical answer path

Feature settings:

- `sentiment_mean_only`
- `sentiment_std_only`
- `sentiment_extreme_only`

Variables:

- `sentiment_mean_lag1`
- `sentiment_std_lag1`
- `sentiment_extreme_lag1`

Model comparison:

- OLS
- Random Forest
- XGBoost

Evaluation:

- MAE, RMSE, and R2
- comparison against historical baseline and macro baseline where relevant
- error diagnostics for stability

Interpretation:

- answers whether institutional sentiment can function as a standalone predictor
- separates sentiment direction, sentiment dispersion, and extreme sentiment share

## 4. RQ3 Alignment

### RQ3

When institutional sentiment representations are added to the macroeconomic baseline, do they improve predictive performance, and how large is their incremental contribution according to delta MAE, delta RMSE, delta R2, and SHAP-based model interpretation?

### Empirical answer path

Reference setting:

- `macro_only`

Candidate macro-sentiment settings:

- `macro_mean`
- `macro_std`
- `macro_extreme`
- `macro_mean_std`
- `macro_mean_extreme`

Evaluation:

- compare each candidate macro-sentiment setting against `macro_only`
- report delta MAE, delta RMSE, and delta R2
- use SHAP to examine model-based predictive contribution
- use error analysis to examine whether gains are stable across market regimes

Delta definitions:

- delta_MAE = MAE(candidate specification) - MAE(macro_only)
- delta_RMSE = RMSE(candidate specification) - RMSE(macro_only)
- delta_R2 = R2(candidate specification) - R2(macro_only)

Interpretation:

- negative delta_MAE means the candidate model has lower prediction error
- negative delta_RMSE means the candidate model has lower prediction error
- positive delta_R2 means the candidate model has higher explanatory performance

## 5. Final Feature Specification Map

| RQ role | Feature setting | Variables | Purpose |
|---|---|---|---|
| RQ1 | `macro_only` | `dgs10_lag1`, `vix_lag1` | Macro baseline |
| RQ2 | `sentiment_mean_only` | `sentiment_mean_lag1` | Standalone sentiment direction |
| RQ2 | `sentiment_std_only` | `sentiment_std_lag1` | Standalone sentiment dispersion |
| RQ2 | `sentiment_extreme_only` | `sentiment_extreme_lag1` | Standalone extreme sentiment share |
| RQ3 | `macro_mean` | macro + `sentiment_mean_lag1` | Sentiment direction beyond macro |
| RQ3 | `macro_std` | macro + `sentiment_std_lag1` | Sentiment dispersion beyond macro |
| RQ3 | `macro_extreme` | macro + `sentiment_extreme_lag1` | Extreme sentiment beyond macro |
| RQ3 | `macro_mean_std` | macro + mean + std | Direction and dispersion beyond macro |
| RQ3 | `macro_mean_extreme` | macro + mean + extreme | Direction and extreme sentiment beyond macro |

## 6. Feature Design Rationale

The final design does not use a single full combined specification as the primary feature setting.

Reason:

- The research questions require separate evidence for macro predictors, standalone sentiment predictors, and incremental sentiment value.
- Sentiment mean, sentiment standard deviation, and extreme sentiment share represent different sentiment dimensions.
- Sentiment standard deviation and extreme sentiment share are conceptually related and empirically highly correlated.
- Separating them improves interpretability, especially for the OLS baseline.
- Limited combined sentiment settings are retained for RQ3.

## 7. Benchmark Position

The proposal mentions the lagged target term as a benchmark predictor.

Final position:

- record its existence as a benchmark idea
- do not include it in the primary feature-setting matrix
- keep the primary design limited to the nine RQ-aligned feature specifications

## 8. Result-to-RQ Mapping

- macro-only model comparison -> answers RQ1
- standalone sentiment model comparison -> answers RQ2
- macro-sentiment delta analysis -> answers RQ3
- SHAP analysis -> supports RQ3 interpretation
- error analysis figures -> support robustness discussion across all RQs

## 9. Writing Guardrails

1. always use daily wording consistently
2. always describe the final target as NASDAQ-100 daily log return
3. do not describe the empirical target as individual AI technology stocks
4. do not collapse the final design back into macro_only / sentiment_only / combined
5. do not interpret standalone sentiment specifications as incremental value beyond macro
6. only macro-sentiment specifications compared against `macro_only` answer RQ3
7. do not present SHAP as causal evidence
8. do not present the lagged target benchmark as a primary feature setting
9. describe nonlinear hyperparameter selection as: lowest CV RMSE, then higher CV R2, then lower grid_order
