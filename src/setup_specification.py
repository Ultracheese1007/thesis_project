from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FEATURE_SPECS = {
    # RQ1
    "macro_only": [
        "dgs10_lag1",
        "vix_lag1"
    ],

    # RQ2
    "sentiment_mean_only": [
        "sentiment_mean_lag1"
    ],
    "sentiment_std_only": [
        "sentiment_std_lag1"
    ],
    "sentiment_extreme_only": [
        "sentiment_extreme_lag1"
    ],

    # RQ3
    "macro_mean": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1"
    ],
    "macro_std": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_std_lag1"
    ],
    "macro_extreme": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_extreme_lag1"
    ],
    "macro_mean_std": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_std_lag1"
    ],
    "macro_mean_extreme": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_extreme_lag1"
    ],
}


RQ_ROLE_MAP = {
    "macro_only": "RQ1_macro_baseline",

    "sentiment_mean_only": "RQ2_standalone_sentiment_direction",
    "sentiment_std_only": "RQ2_standalone_sentiment_dispersion",
    "sentiment_extreme_only": "RQ2_standalone_extreme_sentiment_share",

    "macro_mean": "RQ3_incremental_sentiment_direction_beyond_macro",
    "macro_std": "RQ3_incremental_sentiment_dispersion_beyond_macro",
    "macro_extreme": "RQ3_incremental_extreme_sentiment_beyond_macro",
    "macro_mean_std": "RQ3_incremental_direction_and_dispersion_beyond_macro",
    "macro_mean_extreme": "RQ3_incremental_direction_and_extreme_sentiment_beyond_macro",
}


MODEL_TYPES = ["OLS", "RandomForest", "XGBoost"]


def ensure_directories() -> None:
    required_dirs = [
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "thesis" / "notes",
        PROJECT_ROOT / "data" / "processed",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)


def build_feature_manifest() -> Dict:
    return {
        "project_name": "thesis_project_daily",
        "version": "final_rq_aligned_daily_spec_v1",
        "frequency": "daily",
        "observation_unit": "one trading day",
        "date_column": "date",
        "target": {
            "name": "nasdaq_return",
            "definition": "Daily log return of the NASDAQ-100 index",
            "forecast_timing": "Predict return at day t using predictors available up to day t-1"
        },
        "predictors": {
            "dgs10_lag1": {
                "family": "macro",
                "definition": "10-year U.S. Treasury yield lagged by one trading day",
                "source": "FRED DGS10",
                "lag_days": 1
            },
            "vix_lag1": {
                "family": "macro",
                "definition": "CBOE Volatility Index lagged by one trading day",
                "source": "FRED VIXCLS",
                "lag_days": 1
            },
            "sentiment_mean_lag1": {
                "family": "sentiment",
                "definition": "Daily mean institutional sentiment lagged by one trading day",
                "source": "Daily FinBERT aggregation",
                "lag_days": 1,
                "raw_column_before_lag": "sentiment_mean",
                "interpretation": "Sentiment direction"
            },
            "sentiment_std_lag1": {
                "family": "sentiment",
                "definition": "Daily sentiment standard deviation lagged by one trading day",
                "source": "Daily FinBERT aggregation",
                "lag_days": 1,
                "raw_column_before_lag": "sentiment_std",
                "interpretation": "Sentiment dispersion or disagreement"
            },
            "sentiment_extreme_lag1": {
                "family": "sentiment",
                "definition": "Daily extreme sentiment share/proportion lagged by one trading day",
                "source": "Daily FinBERT aggregation",
                "lag_days": 1,
                "raw_column_before_lag": "sentiment_extreme",
                "interpretation": "Share of highly positive or highly negative news sentiment"
            }
        },
        "primary_feature_settings": FEATURE_SPECS,
        "rq_role_mapping": RQ_ROLE_MAP,
        "feature_design_rationale": {
            "main_design": (
                "The final empirical design uses nine RQ-aligned feature specifications. "
                "This separates the macro baseline, standalone sentiment representations, "
                "and macro-sentiment combinations used to evaluate incremental predictive value."
            ),
            "why_no_single_full_combined_primary_specification": (
                "The thesis does not use one single full combined specification as the primary design "
                "because the goal is to isolate the predictive role of different sentiment representations. "
                "In particular, sentiment_std_lag1 and sentiment_extreme_lag1 are conceptually related "
                "and empirically highly correlated, so separating them improves interpretability, especially "
                "for the linear baseline. The macro_mean_std and macro_mean_extreme specifications are retained "
                "as limited combined sentiment settings for RQ3."
            )
        },
        "lag_policy": {
            "rule": "All predictors are lagged by one trading day.",
            "rationale": [
                "preserve temporal ordering",
                "avoid look-ahead bias",
                "keep macro and sentiment timing consistent"
            ]
        },
        "preprocessing_rules": {
            "standardization": "Fit on training data only and apply fitted parameters to validation/test data.",
            "random_split_allowed": False
        },
        "benchmark_status": {
            "proposal_statement": "The lagged target term is mentioned in the proposal as a benchmark predictor.",
            "included_in_primary_feature_settings": False,
            "final_decision": (
                "The lagged target is documented as a possible benchmark idea, but it is not included "
                "in the final nine RQ-aligned primary feature specifications."
            )
        }
    }


def build_primary_model_comparison_matrix() -> List[Dict[str, str]]:
    matrix: List[Dict[str, str]] = []

    for model_type in MODEL_TYPES:
        for feature_setting, variables in FEATURE_SPECS.items():
            matrix.append(
                {
                    "model_type": model_type,
                    "feature_setting": feature_setting,
                    "rq_role": RQ_ROLE_MAP[feature_setting],
                    "n_predictors": len(variables)
                }
            )

    return matrix


def build_split_manifest() -> Dict:
    return {
        "version": "final_rq_aligned_daily_spec_v1",
        "split_type": "chronological",
        "train_period": {
            "start": "2020-01-03",
            "end": "2023-12-29",
            "n_observations": 1005
        },
        "test_period": {
            "start": "2024-01-02",
            "end": "2024-12-31",
            "n_observations": 252
        },
        "test_role": "strict out-of-sample hold-out set",
        "random_split_allowed": False,
        "cross_validation": {
            "type": "expanding_window_time_series_cv",
            "scope": "training set only",
            "n_folds": 3,
            "shuffle": False,
            "future_information_allowed": False,
            "fold_logic": [
                "Fold 1: earliest training block -> next chronological validation block",
                "Fold 2: expanded training block -> next chronological validation block",
                "Fold 3: further expanded training block -> final chronological validation block before 2024"
            ]
        },
        "preprocessing_scope": {
            "train_test_stage": (
                "Fit preprocessing on the full training set only, then transform the test set "
                "with the fitted training parameters."
            ),
            "cv_stage": (
                "Within each CV fold, fit preprocessing only on the fold-specific training block "
                "and transform the fold-specific validation block."
            )
        },
        "evaluation_metrics": {
            "primary": ["MAE", "RMSE", "R2"],
            "incremental_value_metrics": {
                "delta_MAE": "MAE(candidate_specification) - MAE(macro_only)",
                "delta_RMSE": "RMSE(candidate_specification) - RMSE(macro_only)",
                "delta_R2": "R2(candidate_specification) - R2(macro_only)",
                "interpretation": {
                    "delta_MAE": "Negative values indicate lower error than the macro-only baseline.",
                    "delta_RMSE": "Negative values indicate lower error than the macro-only baseline.",
                    "delta_R2": "Positive values indicate higher explanatory performance than the macro-only baseline."
                },
                "rq3_candidate_specifications": [
                    "macro_mean",
                    "macro_std",
                    "macro_extreme",
                    "macro_mean_std",
                    "macro_mean_extreme"
                ],
                "reference_specification": "macro_only"
            }
        },
        "primary_model_comparison_matrix": build_primary_model_comparison_matrix(),
        "model_selection_rule": {
            "nonlinear_models": ["RandomForest", "XGBoost"],
            "primary_selection_metric": "lowest mean cross-validation RMSE",
            "tie_breaking_rule": [
                "higher mean cross-validation R2",
                "lower grid_order, corresponding to the simpler-first grid ordering"
            ],
            "reported_but_not_selection_metric": [
                "mean cross-validation MAE"
            ],
            "test_set_usage": "The 2024 hold-out test set is not used for hyperparameter tuning."
        },
        "error_analysis_scope": [
            "predicted_vs_actual",
            "residuals_over_time",
            "rolling_window_error_behavior",
            "high_vix_regime_error",
            "extreme_sentiment_regime_error"
        ],
        "interpretability_scope": {
            "ols": "Coefficient signs, magnitude after standardization, and statistical uncertainty are reported cautiously.",
            "tree_based_models": "SHAP is used for model-based predictive contribution, not causal interpretation.",
            "shap_causal_warning": "SHAP values do not establish causal effects of macroeconomic or sentiment variables."
        }
    }


def build_model_info_sheet() -> str:
    return """# Model Info Sheet — Final RQ-Aligned Daily Thesis Specification

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
"""


def build_rq_alignment_notes() -> str:
    return """# RQ Alignment Notes — Final RQ-Aligned Daily Thesis Specification

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
"""


def validate_manifests(feature_manifest: Dict, split_manifest: Dict) -> List[str]:
    errors: List[str] = []

    target_name = feature_manifest.get("target", {}).get("name")
    if target_name != "nasdaq_return":
        errors.append("Target name must be 'nasdaq_return'.")

    predictors = feature_manifest.get("predictors", {})
    predictor_names = set(predictors.keys())

    expected_predictors = {
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_std_lag1",
        "sentiment_extreme_lag1",
    }

    if predictor_names != expected_predictors:
        errors.append(
            "Predictor set must be exactly: "
            "dgs10_lag1, vix_lag1, sentiment_mean_lag1, "
            "sentiment_std_lag1, sentiment_extreme_lag1."
        )

    primary_feature_settings = feature_manifest.get("primary_feature_settings", {})
    expected_feature_settings = set(FEATURE_SPECS.keys())

    if set(primary_feature_settings.keys()) != expected_feature_settings:
        errors.append(
            "Primary feature settings must be exactly the nine RQ-aligned settings: "
            f"{sorted(expected_feature_settings)}."
        )

    for fs_name, expected_cols in FEATURE_SPECS.items():
        actual_cols = primary_feature_settings.get(fs_name)

        if actual_cols != expected_cols:
            errors.append(
                f"Feature setting '{fs_name}' must be exactly {expected_cols}, "
                f"but got {actual_cols}."
            )

        for c in expected_cols:
            if c not in predictor_names:
                errors.append(f"Feature setting '{fs_name}' contains undefined predictor '{c}'.")

    rq_role_mapping = feature_manifest.get("rq_role_mapping", {})
    if set(rq_role_mapping.keys()) != expected_feature_settings:
        errors.append("RQ role mapping must cover all nine feature settings.")

    if split_manifest.get("split_type") != "chronological":
        errors.append("Split type must be chronological.")

    if split_manifest.get("random_split_allowed") is not False:
        errors.append("random_split_allowed must be False.")

    train_period = split_manifest.get("train_period", {})
    test_period = split_manifest.get("test_period", {})

    if train_period.get("start") != "2020-01-03":
        errors.append("Train period start must be 2020-01-03.")

    if train_period.get("end") != "2023-12-29":
        errors.append("Train period end must be 2023-12-29.")

    if test_period.get("start") != "2024-01-02":
        errors.append("Test period start must be 2024-01-02.")

    if test_period.get("end") != "2024-12-31":
        errors.append("Test period end must be 2024-12-31.")

    cv = split_manifest.get("cross_validation", {})
    if cv.get("type") != "expanding_window_time_series_cv":
        errors.append("CV type must be expanding_window_time_series_cv.")

    if cv.get("n_folds") != 3:
        errors.append("CV n_folds must be 3.")

    if cv.get("shuffle") is not False:
        errors.append("CV shuffle must be False.")

    if cv.get("future_information_allowed") is not False:
        errors.append("future_information_allowed must be False.")

    primary_metrics = split_manifest.get("evaluation_metrics", {}).get("primary", [])
    if set(primary_metrics) != {"MAE", "RMSE", "R2"}:
        errors.append("Primary evaluation metrics must be exactly MAE, RMSE, and R2.")

    incremental = split_manifest.get("evaluation_metrics", {}).get("incremental_value_metrics", {})
    expected_rq3_specs = {
        "macro_mean",
        "macro_std",
        "macro_extreme",
        "macro_mean_std",
        "macro_mean_extreme"
    }

    actual_rq3_specs = set(incremental.get("rq3_candidate_specifications", []))
    if actual_rq3_specs != expected_rq3_specs:
        errors.append(
            "RQ3 candidate specifications must be exactly: "
            "macro_mean, macro_std, macro_extreme, macro_mean_std, macro_mean_extreme."
        )

    if incremental.get("reference_specification") != "macro_only":
        errors.append("RQ3 reference specification must be macro_only.")

    selection_rule = split_manifest.get("model_selection_rule", {})
    expected_tie_rule = [
        "higher mean cross-validation R2",
        "lower grid_order, corresponding to the simpler-first grid ordering"
    ]

    if selection_rule.get("primary_selection_metric") != "lowest mean cross-validation RMSE":
        errors.append("Primary selection metric must be lowest mean cross-validation RMSE.")

    if selection_rule.get("tie_breaking_rule") != expected_tie_rule:
        errors.append("Tie-breaking rule must be: higher CV R2, then lower grid_order.")

    error_scope = set(split_manifest.get("error_analysis_scope", []))
    required_error_scope = {
        "predicted_vs_actual",
        "residuals_over_time",
        "rolling_window_error_behavior",
        "high_vix_regime_error",
        "extreme_sentiment_regime_error"
    }

    missing = required_error_scope - error_scope
    if missing:
        errors.append(f"Missing required error analysis items: {sorted(missing)}")

    matrix = split_manifest.get("primary_model_comparison_matrix", [])
    expected_matrix_size = len(MODEL_TYPES) * len(FEATURE_SPECS)

    if len(matrix) != expected_matrix_size:
        errors.append(
            f"Primary model comparison matrix must contain {expected_matrix_size} "
            "model-setting combinations."
        )

    observed_pairs = {
        (row.get("model_type"), row.get("feature_setting"))
        for row in matrix
    }

    expected_pairs = {
        (model_type, feature_setting)
        for model_type in MODEL_TYPES
        for feature_setting in FEATURE_SPECS.keys()
    }

    if observed_pairs != expected_pairs:
        errors.append("Primary model comparison matrix does not match 3 models × 9 feature settings.")

    return errors


def write_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()

    feature_manifest = build_feature_manifest()
    split_manifest = build_split_manifest()
    model_info_sheet = build_model_info_sheet()
    rq_alignment_notes = build_rq_alignment_notes()

    errors = validate_manifests(feature_manifest, split_manifest)
    if errors:
        print("Validation failed:")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(1)

    write_json(PROJECT_ROOT / "data" / "processed" / "feature_manifest.json", feature_manifest)
    write_json(PROJECT_ROOT / "data" / "processed" / "split_manifest.json", split_manifest)
    write_text(PROJECT_ROOT / "docs" / "model_info_sheet.md", model_info_sheet)
    write_text(PROJECT_ROOT / "thesis" / "notes" / "rq_alignment_notes.md", rq_alignment_notes)

    print("Final RQ-aligned specification files generated successfully:")
    print(" - docs/model_info_sheet.md")
    print(" - data/processed/feature_manifest.json")
    print(" - data/processed/split_manifest.json")
    print(" - thesis/notes/rq_alignment_notes.md")


if __name__ == "__main__":
    main()