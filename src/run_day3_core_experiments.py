from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 0. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"
RESULTS_COMPARISON_DIR = RESULTS_DIR / "comparison"
RESULTS_ERROR_DIR = RESULTS_DIR / "error_analysis"
RESULTS_SHAP_DIR = RESULTS_DIR / "shap"

FINAL_DATASET_PATH = DATA_PROCESSED_DIR / "final_daily_dataset.csv"
TRAIN_PATH = DATA_PROCESSED_DIR / "train_daily.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test_daily.csv"

DATE_COL = "date"
TARGET_COL = "nasdaq_return"


# ============================================================
# 1. Experiment design
# ============================================================

FEATURE_SPECS = {
    # RQ1: macroeconomic predictive power
    "macro_only": [
        "dgs10_lag1",
        "vix_lag1",
    ],

    # RQ2: standalone sentiment predictive power
    "sentiment_mean_only": [
        "sentiment_mean_lag1",
    ],
    "sentiment_std_only": [
        "sentiment_std_lag1",
    ],
    "sentiment_extreme_only": [
        "sentiment_extreme_lag1",
    ],

    # RQ3: incremental sentiment value beyond macro baseline
    "macro_mean": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
    ],
    "macro_std": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_std_lag1",
    ],
    "macro_extreme": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_extreme_lag1",
    ],
    "macro_mean_std": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_std_lag1",
    ],
    "macro_mean_extreme": [
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_extreme_lag1",
    ],
}


FEATURE_SPEC_ROLES = {
    "macro_only": "RQ1_macro_baseline",

    "sentiment_mean_only": "RQ2_standalone_sentiment",
    "sentiment_std_only": "RQ2_standalone_sentiment",
    "sentiment_extreme_only": "RQ2_standalone_sentiment",

    "macro_mean": "RQ3_incremental_sentiment_beyond_macro",
    "macro_std": "RQ3_incremental_sentiment_beyond_macro",
    "macro_extreme": "RQ3_incremental_sentiment_beyond_macro",
    "macro_mean_std": "RQ3_incremental_sentiment_beyond_macro",
    "macro_mean_extreme": "RQ3_incremental_sentiment_beyond_macro",
}


RQ3_INCREMENTAL_SPECS = {
    "macro_mean",
    "macro_std",
    "macro_extreme",
    "macro_mean_std",
    "macro_mean_extreme",
}


N_SPLITS = 3
ROLLING_WINDOW = 20
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# 2. Utility functions
# ============================================================

def ensure_dirs() -> None:
    dirs = [
        RESULTS_MODELS_DIR / "ols" / "cv",
        RESULTS_MODELS_DIR / "ols" / "predictions",
        RESULTS_MODELS_DIR / "ols" / "coefficients",
        RESULTS_MODELS_DIR / "ols" / "summaries",

        RESULTS_MODELS_DIR / "random_forest" / "cv",
        RESULTS_MODELS_DIR / "random_forest" / "predictions",
        RESULTS_MODELS_DIR / "random_forest" / "feature_importance",

        RESULTS_MODELS_DIR / "xgboost" / "cv",
        RESULTS_MODELS_DIR / "xgboost" / "predictions",
        RESULTS_MODELS_DIR / "xgboost" / "feature_importance",

        RESULTS_COMPARISON_DIR,
        RESULTS_ERROR_DIR,

        RESULTS_SHAP_DIR / "random_forest",
        RESULTS_SHAP_DIR / "xgboost",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_df = pd.read_csv(FINAL_DATASET_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in [final_df, train_df, test_df]:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    return final_df, train_df, test_df


def validate_columns(df: pd.DataFrame) -> None:
    required = {DATE_COL, TARGET_COL}
    for cols in FEATURE_SPECS.values():
        required.update(cols)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.var(y_true) == 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": safe_rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
    }


def make_tscv(n_splits: int = N_SPLITS) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_other_scaled = pd.DataFrame(
        scaler.transform(X_other),
        columns=X_other.columns,
        index=X_other.index,
    )
    return X_train_scaled, X_other_scaled


def ols_condition_number(X_scaled: pd.DataFrame) -> float:
    X_const = sm.add_constant(X_scaled, has_constant="add")
    return float(np.linalg.cond(X_const.values))


def build_model(model_family: str):
    if model_family == "ols":
        return None
    if model_family == "random_forest":
        return RandomForestRegressor(**RF_PARAMS)
    if model_family == "xgboost":
        return XGBRegressor(**XGB_PARAMS)
    raise ValueError(f"Unsupported model family: {model_family}")


# ============================================================
# 3. CV runners
# ============================================================

def run_cv_for_spec(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    spec_name: str,
) -> tuple[pd.DataFrame, dict]:
    X_all, y_all = prepare_xy(train_df, feature_cols)
    tscv = make_tscv()

    fold_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(tscv.split(X_all), start=1):
        X_tr = X_all.iloc[train_idx].copy()
        y_tr = y_all.iloc[train_idx].copy()
        X_val = X_all.iloc[val_idx].copy()
        y_val = y_all.iloc[val_idx].copy()

        scaler = fit_scaler(X_tr)
        X_tr_scaled, X_val_scaled = transform_features(scaler, X_tr, X_val)

        if model_family == "ols":
            X_tr_const = sm.add_constant(X_tr_scaled, has_constant="add")
            X_val_const = sm.add_constant(X_val_scaled, has_constant="add")

            model = sm.OLS(y_tr, X_tr_const).fit()
            val_pred = model.predict(X_val_const)

        else:
            model = build_model(model_family)
            model.fit(X_tr_scaled, y_tr)
            val_pred = model.predict(X_val_scaled)

        fold_metrics = calc_metrics(y_val.values, np.asarray(val_pred))

        fold_rows.append({
            "model_family": model_family,
            "spec": spec_name,
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_start_date": train_df.iloc[val_idx][DATE_COL].min(),
            "val_end_date": train_df.iloc[val_idx][DATE_COL].max(),
            "cv_mae": fold_metrics["mae"],
            "cv_rmse": fold_metrics["rmse"],
            "cv_r2": fold_metrics["r2"],
        })

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "cv_mean_mae": fold_df["cv_mae"].mean(),
        "cv_std_mae": fold_df["cv_mae"].std(ddof=1),
        "cv_mean_rmse": fold_df["cv_rmse"].mean(),
        "cv_std_rmse": fold_df["cv_rmse"].std(ddof=1),
        "cv_mean_r2": fold_df["cv_r2"].mean(),
        "cv_std_r2": fold_df["cv_r2"].std(ddof=1),
    }

    return fold_df, summary


# ============================================================
# 4. Final fit + prediction export
# ============================================================

def run_final_fit_for_spec(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    spec_name: str,
) -> dict:
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    scaler = fit_scaler(X_train)
    X_train_scaled, X_test_scaled = transform_features(scaler, X_train, X_test)

    if model_family == "ols":
        X_train_const = sm.add_constant(X_train_scaled, has_constant="add")
        X_test_const = sm.add_constant(X_test_scaled, has_constant="add")

        model = sm.OLS(y_train, X_train_const).fit()

        train_pred = np.asarray(model.predict(X_train_const))
        test_pred = np.asarray(model.predict(X_test_const))

        coef_table = pd.DataFrame({
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
        })
        coef_table["model_family"] = model_family
        coef_table["spec"] = spec_name
        coef_table["condition_number"] = ols_condition_number(X_train_scaled)

        coef_out = RESULTS_MODELS_DIR / "ols" / "coefficients" / f"{spec_name}_coefficients.csv"
        coef_table.to_csv(coef_out, index=False)

        summary_txt_path = RESULTS_MODELS_DIR / "ols" / "summaries" / f"{spec_name}_ols_summary.txt"
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

    else:
        model = build_model(model_family)
        model.fit(X_train_scaled, y_train)

        train_pred = np.asarray(model.predict(X_train_scaled))
        test_pred = np.asarray(model.predict(X_test_scaled))

        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
            "model_family": model_family,
            "spec": spec_name,
        }).sort_values("importance", ascending=False)

        fi_dir = RESULTS_MODELS_DIR / model_family / "feature_importance"
        fi_dir.mkdir(parents=True, exist_ok=True)
        feature_importance_df.to_csv(fi_dir / f"{spec_name}_feature_importance.csv", index=False)

        shap_dir = RESULTS_SHAP_DIR / model_family
        shap_dir.mkdir(parents=True, exist_ok=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        shap_long_df = pd.DataFrame({
            DATE_COL: test_df[DATE_COL].values.repeat(len(feature_cols)),
            "feature": np.tile(feature_cols, len(X_test_scaled)),
            "feature_value": X_test_scaled.values.reshape(-1),
            "shap_value": np.asarray(shap_values).reshape(-1),
            "model_family": model_family,
            "spec": spec_name,
        })

        shap_long_df.to_csv(shap_dir / f"{spec_name}_test_shap_values.csv", index=False)

        plt.figure()
        shap.summary_plot(shap_values, X_test_scaled, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{spec_name}_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()

    train_metrics = calc_metrics(y_train.values, train_pred)
    test_metrics = calc_metrics(y_test.values, test_pred)

    train_pred_df = pd.DataFrame({
        DATE_COL: train_df[DATE_COL],
        "actual": y_train.values,
        "predicted": train_pred,
        "residual": y_train.values - train_pred,
        "abs_error": np.abs(y_train.values - train_pred),
        "squared_error": (y_train.values - train_pred) ** 2,
        "model_family": model_family,
        "spec": spec_name,
        "dataset": "train",
    })

    test_pred_df = pd.DataFrame({
        DATE_COL: test_df[DATE_COL],
        "actual": y_test.values,
        "predicted": test_pred,
        "residual": y_test.values - test_pred,
        "abs_error": np.abs(y_test.values - test_pred),
        "squared_error": (y_test.values - test_pred) ** 2,
        "model_family": model_family,
        "spec": spec_name,
        "dataset": "test",
    })

    pred_dir = RESULTS_MODELS_DIR / model_family / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    train_pred_df.to_csv(pred_dir / f"{spec_name}_train_predictions.csv", index=False)
    test_pred_df.to_csv(pred_dir / f"{spec_name}_test_predictions.csv", index=False)

    result = {
        "model": model,
        "scaler": scaler,
        "train_pred_df": train_pred_df,
        "test_pred_df": test_pred_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    return result


# ============================================================
# 5. Comparison tables
# ============================================================

def build_delta_metrics_table(master_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build delta metrics relative to macro_only within each model family.

    Important interpretation:
    - For RQ3 macro-sentiment specifications, delta metrics measure incremental
      sentiment value beyond the macro baseline.
    - For RQ2 standalone sentiment specifications, delta metrics only compare
      standalone sentiment performance against macro_only as a reference. They
      should not be interpreted as incremental value, because these specs do not
      add sentiment to the macro baseline.
    """

    baselines = master_metrics_df[master_metrics_df["spec"] == "macro_only"].copy()
    baselines = baselines[[
        "model_family",
        "cv_mean_mae",
        "cv_mean_rmse",
        "cv_mean_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
    ]].rename(columns={
        "cv_mean_mae": "baseline_cv_mean_mae",
        "cv_mean_rmse": "baseline_cv_mean_rmse",
        "cv_mean_r2": "baseline_cv_mean_r2",
        "test_mae": "baseline_test_mae",
        "test_rmse": "baseline_test_rmse",
        "test_r2": "baseline_test_r2",
    })

    delta_df = master_metrics_df.merge(baselines, on="model_family", how="left")

    delta_df["feature_spec_role"] = delta_df["spec"].map(FEATURE_SPEC_ROLES)
    delta_df["is_rq3_incremental_spec"] = delta_df["spec"].isin(RQ3_INCREMENTAL_SPECS)

    delta_df["delta_cv_mae_vs_macro_only"] = (
        delta_df["cv_mean_mae"] - delta_df["baseline_cv_mean_mae"]
    )
    delta_df["delta_cv_rmse_vs_macro_only"] = (
        delta_df["cv_mean_rmse"] - delta_df["baseline_cv_mean_rmse"]
    )
    delta_df["delta_cv_r2_vs_macro_only"] = (
        delta_df["cv_mean_r2"] - delta_df["baseline_cv_mean_r2"]
    )

    delta_df["delta_test_mae_vs_macro_only"] = (
        delta_df["test_mae"] - delta_df["baseline_test_mae"]
    )
    delta_df["delta_test_rmse_vs_macro_only"] = (
        delta_df["test_rmse"] - delta_df["baseline_test_rmse"]
    )
    delta_df["delta_test_r2_vs_macro_only"] = (
        delta_df["test_r2"] - delta_df["baseline_test_r2"]
    )

    delta_df["delta_error_interpretation"] = (
        "For MAE/RMSE deltas, negative values indicate lower error than macro_only. "
        "For R2 deltas, positive values indicate higher R2 than macro_only."
    )

    delta_df["comparison_note"] = np.where(
        delta_df["is_rq3_incremental_spec"],
        "RQ3: incremental sentiment value beyond macro_only",
        np.where(
            delta_df["spec"] == "macro_only",
            "RQ1: macro baseline reference",
            "RQ2: standalone sentiment performance compared against macro_only reference; not an incremental-addition test"
        )
    )

    return delta_df


def build_model_ranking_table(master_metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = master_metrics_df.copy()
    ranking_df["rank_test_rmse"] = ranking_df["test_rmse"].rank(method="dense", ascending=True)
    ranking_df["rank_test_mae"] = ranking_df["test_mae"].rank(method="dense", ascending=True)
    ranking_df["rank_test_r2"] = ranking_df["test_r2"].rank(method="dense", ascending=False)

    ranking_df = ranking_df.sort_values(
        by=["rank_test_rmse", "rank_test_mae", "rank_test_r2", "model_family", "spec"]
    ).reset_index(drop=True)

    return ranking_df


# ============================================================
# 6. Error analysis
# ============================================================

def build_error_analysis_outputs(
    train_df: pd.DataFrame,
    all_test_pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high_vix_threshold = train_df["vix_lag1"].quantile(0.75)
    high_extreme_sentiment_threshold = train_df["sentiment_extreme_lag1"].quantile(0.75)

    test_regime_cols = all_test_pred_df.copy()

    regime_source = train_test_concat_for_regimes()
    regime_source = regime_source[[DATE_COL, "vix_lag1", "sentiment_extreme_lag1"]].drop_duplicates()

    test_regime_cols = test_regime_cols.merge(regime_source, on=DATE_COL, how="left")

    test_regime_cols["vix_regime"] = np.where(
        test_regime_cols["vix_lag1"] >= high_vix_threshold,
        "high_vix",
        "normal_vix",
    )

    test_regime_cols["sentiment_regime"] = np.where(
        test_regime_cols["sentiment_extreme_lag1"] >= high_extreme_sentiment_threshold,
        "high_extreme_sentiment",
        "normal_extreme_sentiment",
    )

    residuals_over_time = test_regime_cols.copy()

    regime_rows = []

    for (model_family, spec, regime_label), g in test_regime_cols.groupby(["model_family", "spec", "vix_regime"]):
        regime_rows.append({
            "model_family": model_family,
            "spec": spec,
            "regime_type": "vix_regime",
            "regime_label": regime_label,
            "n_obs": len(g),
            "mae": g["abs_error"].mean(),
            "rmse": float(np.sqrt(g["squared_error"].mean())),
            "r2": safe_r2(g["actual"].values, g["predicted"].values),
        })

    for (model_family, spec, regime_label), g in test_regime_cols.groupby(["model_family", "spec", "sentiment_regime"]):
        regime_rows.append({
            "model_family": model_family,
            "spec": spec,
            "regime_type": "sentiment_regime",
            "regime_label": regime_label,
            "n_obs": len(g),
            "mae": g["abs_error"].mean(),
            "rmse": float(np.sqrt(g["squared_error"].mean())),
            "r2": safe_r2(g["actual"].values, g["predicted"].values),
        })

    regime_error_summary = pd.DataFrame(regime_rows)

    rolling_df = test_regime_cols.sort_values(["model_family", "spec", DATE_COL]).copy()

    rolling_parts = []
    for (model_family, spec), g in rolling_df.groupby(["model_family", "spec"]):
        g = g.sort_values(DATE_COL).copy()
        g["rolling_mae_20"] = g["abs_error"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        g["rolling_rmse_20"] = np.sqrt(g["squared_error"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean())
        g["rolling_mean_residual_20"] = g["residual"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        rolling_parts.append(g)

    rolling_error_summary = pd.concat(rolling_parts, ignore_index=True)

    return residuals_over_time, regime_error_summary, rolling_error_summary


def train_test_concat_for_regimes() -> pd.DataFrame:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])
    return pd.concat([train_df, test_df], ignore_index=True)


# ============================================================
# 7. Main pipeline
# ============================================================

def main() -> None:
    ensure_dirs()

    final_df, train_df, test_df = load_data()
    validate_columns(final_df)

    experiment_manifest = {
        "dataset": {
            "final_dataset_path": str(FINAL_DATASET_PATH),
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
            "train_start": str(train_df[DATE_COL].min().date()),
            "train_end": str(train_df[DATE_COL].max().date()),
            "test_start": str(test_df[DATE_COL].min().date()),
            "test_end": str(test_df[DATE_COL].max().date()),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        },
        "target": TARGET_COL,
        "date_col": DATE_COL,
        "feature_specs": FEATURE_SPECS,
        "feature_spec_roles": FEATURE_SPEC_ROLES,
        "rq_mapping": {
            "RQ1": {
                "purpose": "macroeconomic predictive power",
                "feature_settings": ["macro_only"],
            },
            "RQ2": {
                "purpose": "standalone sentiment predictive power",
                "feature_settings": [
                    "sentiment_mean_only",
                    "sentiment_std_only",
                    "sentiment_extreme_only",
                ],
                "interpretation_note": (
                    "These specifications test each FinBERT-based sentiment representation "
                    "as a standalone predictor. They are not full sentiment-combination models."
                ),
            },
            "RQ3": {
                "purpose": "incremental sentiment value beyond macro baseline",
                "feature_settings": [
                    "macro_mean",
                    "macro_std",
                    "macro_extreme",
                    "macro_mean_std",
                    "macro_mean_extreme",
                ],
                "reference_setting": "macro_only",
            },
        },
        "collinearity_design_note": (
            "The design does not use a full combined specification containing sentiment_mean_lag1, "
            "sentiment_std_lag1, and sentiment_extreme_lag1 together with macro predictors as a primary "
            "setting, because sentiment_std_lag1 and sentiment_extreme_lag1 show high pairwise correlation "
            "and may capture overlapping sentiment-intensity information."
        ),
        "cv": {
            "type": "expanding_window",
            "n_splits": N_SPLITS,
        },
        "rolling_error_window": ROLLING_WINDOW,
        "random_state": RANDOM_STATE,
        "rf_params": RF_PARAMS,
        "xgb_params": XGB_PARAMS,
    }

    with open(RESULTS_COMPARISON_DIR / "experiment_manifest_day3.json", "w", encoding="utf-8") as f:
        json.dump(experiment_manifest, f, indent=2, default=str)

    model_families = ["ols", "random_forest", "xgboost"]

    master_rows = []
    all_test_predictions = []

    for model_family in model_families:
        for spec_name, feature_cols in FEATURE_SPECS.items():
            print(f"Running {model_family} | {spec_name} ...")

            cv_fold_df, cv_summary = run_cv_for_spec(
                train_df=train_df,
                feature_cols=feature_cols,
                model_family=model_family,
                spec_name=spec_name,
            )

            cv_dir = RESULTS_MODELS_DIR / model_family / "cv"
            cv_fold_df.to_csv(cv_dir / f"{spec_name}_cv_fold_metrics.csv", index=False)

            final_result = run_final_fit_for_spec(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                model_family=model_family,
                spec_name=spec_name,
            )

            all_test_predictions.append(final_result["test_pred_df"])

            row = {
                "model_family": model_family,
                "spec": spec_name,
                "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                "n_features": len(feature_cols),
                "feature_list": ", ".join(feature_cols),

                "cv_mean_mae": cv_summary["cv_mean_mae"],
                "cv_std_mae": cv_summary["cv_std_mae"],
                "cv_mean_rmse": cv_summary["cv_mean_rmse"],
                "cv_std_rmse": cv_summary["cv_std_rmse"],
                "cv_mean_r2": cv_summary["cv_mean_r2"],
                "cv_std_r2": cv_summary["cv_std_r2"],

                "train_mae": final_result["train_metrics"]["mae"],
                "train_rmse": final_result["train_metrics"]["rmse"],
                "train_r2": final_result["train_metrics"]["r2"],

                "test_mae": final_result["test_metrics"]["mae"],
                "test_rmse": final_result["test_metrics"]["rmse"],
                "test_r2": final_result["test_metrics"]["r2"],
            }

            master_rows.append(row)

    master_metrics_df = pd.DataFrame(master_rows)
    master_metrics_df = master_metrics_df.sort_values(["model_family", "spec"]).reset_index(drop=True)
    master_metrics_df.to_csv(RESULTS_COMPARISON_DIR / "master_metrics_table.csv", index=False)

    delta_metrics_df = build_delta_metrics_table(master_metrics_df)
    delta_metrics_df.to_csv(RESULTS_COMPARISON_DIR / "delta_metrics_table.csv", index=False)

    ranking_df = build_model_ranking_table(master_metrics_df)
    ranking_df.to_csv(RESULTS_COMPARISON_DIR / "model_ranking_table.csv", index=False)

    all_test_pred_df = pd.concat(all_test_predictions, ignore_index=True)
    residuals_over_time, regime_error_summary, rolling_error_summary = build_error_analysis_outputs(
        train_df=train_df,
        all_test_pred_df=all_test_pred_df,
    )

    residuals_over_time.to_csv(RESULTS_ERROR_DIR / "residuals_over_time.csv", index=False)
    regime_error_summary.to_csv(RESULTS_ERROR_DIR / "regime_error_summary.csv", index=False)
    rolling_error_summary.to_csv(RESULTS_ERROR_DIR / "rolling_error_summary.csv", index=False)

    print("\n=== Day 3 core experiments completed successfully ===")
    print(f"Master metrics: {RESULTS_COMPARISON_DIR / 'master_metrics_table.csv'}")
    print(f"Delta metrics: {RESULTS_COMPARISON_DIR / 'delta_metrics_table.csv'}")
    print(f"Ranking table: {RESULTS_COMPARISON_DIR / 'model_ranking_table.csv'}")
    print(f"Residuals over time: {RESULTS_ERROR_DIR / 'residuals_over_time.csv'}")
    print(f"Regime error summary: {RESULTS_ERROR_DIR / 'regime_error_summary.csv'}")
    print(f"Rolling error summary: {RESULTS_ERROR_DIR / 'rolling_error_summary.csv'}")


if __name__ == "__main__":
    main()