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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 0. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_PROCESSED_DIR / "train_daily.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test_daily.csv"

TUNING_DIR = RESULTS_DIR / "tuning"
TUNING_COMPARISON_DIR = TUNING_DIR / "comparison"

TUNED_MASTER_PATH = TUNING_COMPARISON_DIR / "tuned_nonlinear_master_metrics.csv"
TUNED_BEST_PARAMS_PATH = TUNING_COMPARISON_DIR / "tuned_best_params_summary.csv"

ALL_MODELS_DIR = TUNING_DIR / "all_models"
ALL_ERROR_DIR = ALL_MODELS_DIR / "error_analysis"
ALL_SHAP_DIR = ALL_MODELS_DIR / "shap"
ALL_SHAP_VALUES_DIR = ALL_SHAP_DIR / "values"
ALL_SHAP_PLOTS_DIR = ALL_SHAP_DIR / "plots"

DATE_COL = "date"
TARGET_COL = "nasdaq_return"

ROLLING_WINDOW = 20
RANDOM_STATE = 42


# ============================================================
# 1. Feature specifications
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


NONLINEAR_MODELS = ["random_forest", "xgboost"]


XGB_FIXED_PARAMS = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# 2. Utilities
# ============================================================

def ensure_dirs() -> None:
    dirs = [
        TUNING_COMPARISON_DIR,
        ALL_ERROR_DIR,
        ALL_SHAP_DIR,
        ALL_SHAP_VALUES_DIR,
        ALL_SHAP_PLOTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in (train_df, test_df):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    return train_df, test_df


def validate_columns(df: pd.DataFrame) -> None:
    required = {DATE_COL, TARGET_COL}

    for feature_cols in FEATURE_SPECS.values():
        required.update(feature_cols)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


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


def as_int_if_present(row: pd.Series, col: str) -> int | None:
    if col not in row.index:
        return None
    value = row[col]
    if pd.isna(value):
        return None
    return int(value)


def as_float_if_present(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    value = row[col]
    if pd.isna(value):
        return None
    return float(value)


def get_best_params(
    best_params_df: pd.DataFrame,
    model_family: str,
    spec_name: str,
) -> dict:
    row_df = best_params_df[
        (best_params_df["model_family"] == model_family)
        & (best_params_df["spec"] == spec_name)
    ].copy()

    if row_df.empty:
        raise ValueError(f"No tuned best params found for {model_family} | {spec_name}")

    row = row_df.iloc[0]

    if model_family == "random_forest":
        params = {
            "n_estimators": as_int_if_present(row, "n_estimators"),
            "max_depth": as_int_if_present(row, "max_depth"),
            "min_samples_leaf": as_int_if_present(row, "min_samples_leaf"),
        }
        return {k: v for k, v in params.items() if v is not None}

    if model_family == "xgboost":
        params = {
            "n_estimators": as_int_if_present(row, "n_estimators"),
            "learning_rate": as_float_if_present(row, "learning_rate"),
            "max_depth": as_int_if_present(row, "max_depth"),
        }
        return {k: v for k, v in params.items() if v is not None}

    raise ValueError(f"Unsupported model family: {model_family}")


def build_model(model_family: str, best_params: dict):
    if model_family == "random_forest":
        return RandomForestRegressor(
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_family == "xgboost":
        full_params = {
            **best_params,
            **XGB_FIXED_PARAMS,
        }
        return XGBRegressor(**full_params)

    raise ValueError(f"Unsupported model family: {model_family}")


def normalize_shap_values(shap_values: np.ndarray, n_features: int) -> np.ndarray:
    arr = np.asarray(shap_values)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim == 3:
        # Defensive handling for possible multi-output return shape.
        # For regression this should normally not happen, but if it does,
        # use the first output dimension.
        arr = arr[:, :, 0]

    if arr.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP shape {arr.shape}; expected second dimension = {n_features}."
        )

    return arr


# ============================================================
# 3. Tuned nonlinear delta table
# ============================================================

def build_tuned_delta_vs_macro_only(tuned_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RQ3-only delta table for tuned nonlinear models.

    Only macro-sentiment specifications are interpreted as incremental sentiment
    value beyond the macro baseline.

    RQ2 standalone sentiment specifications are deliberately excluded from this
    table because they do not add sentiment to macro_only.
    """

    df = tuned_master_df[
        tuned_master_df["model_family"].isin(NONLINEAR_MODELS)
    ].copy()

    baselines = df[df["spec"] == "macro_only"].copy()
    baselines = baselines[[
        "model_family",
        "cv_mean_mae",
        "cv_mean_rmse",
        "cv_mean_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
    ]].rename(columns={
        "cv_mean_mae": "macro_only_cv_mean_mae",
        "cv_mean_rmse": "macro_only_cv_mean_rmse",
        "cv_mean_r2": "macro_only_cv_mean_r2",
        "test_mae": "macro_only_test_mae",
        "test_rmse": "macro_only_test_rmse",
        "test_r2": "macro_only_test_r2",
    })

    rq3_df = df[df["spec"].isin(RQ3_INCREMENTAL_SPECS)].copy()
    delta_df = rq3_df.merge(baselines, on="model_family", how="left")

    delta_df["feature_spec_role"] = delta_df["spec"].map(FEATURE_SPEC_ROLES)

    delta_df["delta_cv_mae_vs_macro_only"] = (
        delta_df["cv_mean_mae"] - delta_df["macro_only_cv_mean_mae"]
    )
    delta_df["delta_cv_rmse_vs_macro_only"] = (
        delta_df["cv_mean_rmse"] - delta_df["macro_only_cv_mean_rmse"]
    )
    delta_df["delta_cv_r2_vs_macro_only"] = (
        delta_df["cv_mean_r2"] - delta_df["macro_only_cv_mean_r2"]
    )

    delta_df["delta_test_mae_vs_macro_only"] = (
        delta_df["test_mae"] - delta_df["macro_only_test_mae"]
    )
    delta_df["delta_test_rmse_vs_macro_only"] = (
        delta_df["test_rmse"] - delta_df["macro_only_test_rmse"]
    )
    delta_df["delta_test_r2_vs_macro_only"] = (
        delta_df["test_r2"] - delta_df["macro_only_test_r2"]
    )

    delta_df["delta_interpretation"] = (
        "For MAE/RMSE deltas, negative values indicate lower error than macro_only. "
        "For R2 deltas, positive values indicate higher R2 than macro_only."
    )

    sort_cols = ["model_family", "spec"]
    return delta_df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================
# 4. Refit tuned models, predictions, SHAP
# ============================================================

def refit_predict_and_shap_all(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    shap_value_parts = []
    mean_abs_shap_rows = []

    for model_family in NONLINEAR_MODELS:
        for spec_name, feature_cols in FEATURE_SPECS.items():
            print(f"\n=== Refit tuned model + SHAP | {model_family} | {spec_name} ===")

            best_params = get_best_params(
                best_params_df=best_params_df,
                model_family=model_family,
                spec_name=spec_name,
            )

            X_train, y_train = prepare_xy(train_df, feature_cols)
            X_test, y_test = prepare_xy(test_df, feature_cols)

            scaler = fit_scaler(X_train)
            X_train_scaled, X_test_scaled = transform_features(
                scaler=scaler,
                X_train=X_train,
                X_other=X_test,
            )

            model = build_model(model_family, best_params)
            model.fit(X_train_scaled, y_train)

            train_pred = np.asarray(model.predict(X_train_scaled))
            test_pred = np.asarray(model.predict(X_test_scaled))

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
                "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                "dataset": "train",
                "tuned": True,
                "n_features": len(feature_cols),
                "feature_list": ", ".join(feature_cols),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "best_params_json": json.dumps(best_params),
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
                "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                "dataset": "test",
                "tuned": True,
                "n_features": len(feature_cols),
                "feature_list": ", ".join(feature_cols),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "best_params_json": json.dumps(best_params),
            })

            prediction_parts.append(train_pred_df)
            prediction_parts.append(test_pred_df)

            # -------------------------
            # SHAP values on hold-out test set
            # -------------------------
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            shap_arr = normalize_shap_values(shap_values, n_features=len(feature_cols))

            shap_long_df = pd.DataFrame({
                DATE_COL: np.repeat(test_df[DATE_COL].values, len(feature_cols)),
                "model_family": model_family,
                "spec": spec_name,
                "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                "feature": np.tile(feature_cols, len(X_test_scaled)),
                "feature_value_scaled": X_test_scaled.values.reshape(-1),
                "shap_value": shap_arr.reshape(-1),
                "abs_shap_value": np.abs(shap_arr.reshape(-1)),
                "dataset": "test",
                "tuned": True,
                "best_params_json": json.dumps(best_params),
            })

            shap_value_parts.append(shap_long_df)

            individual_shap_path = (
                ALL_SHAP_VALUES_DIR
                / f"{model_family}_{spec_name}_test_shap_values.csv"
            )
            shap_long_df.to_csv(individual_shap_path, index=False)

            for j, feature in enumerate(feature_cols):
                mean_abs_shap_rows.append({
                    "model_family": model_family,
                    "spec": spec_name,
                    "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                    "feature": feature,
                    "mean_abs_shap": float(np.mean(np.abs(shap_arr[:, j]))),
                    "mean_shap": float(np.mean(shap_arr[:, j])),
                    "std_shap": float(np.std(shap_arr[:, j], ddof=1)),
                    "mean_feature_value_scaled": float(X_test_scaled[feature].mean()),
                    "std_feature_value_scaled": float(X_test_scaled[feature].std(ddof=1)),
                    "n_test": int(len(X_test_scaled)),
                    "n_features": len(feature_cols),
                    "best_params_json": json.dumps(best_params),
                })

            # -------------------------
            # SHAP summary plot
            # -------------------------
            plot_path = ALL_SHAP_PLOTS_DIR / f"{model_family}_{spec_name}_shap_summary.png"

            plt.figure()
            shap.summary_plot(
                shap_arr,
                X_test_scaled,
                show=False,
                plot_size=(8, 5),
            )
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()

    predictions_all_df = pd.concat(prediction_parts, ignore_index=True)
    shap_values_all_df = pd.concat(shap_value_parts, ignore_index=True)

    mean_abs_shap_df = pd.DataFrame(mean_abs_shap_rows)
    mean_abs_shap_df = mean_abs_shap_df.sort_values(
        ["model_family", "spec", "mean_abs_shap"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return predictions_all_df, shap_values_all_df, mean_abs_shap_df


# ============================================================
# 5. Error analysis
# ============================================================

def build_error_analysis_outputs(
    train_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high_vix_threshold = train_df["vix_lag1"].quantile(0.75)
    high_extreme_sentiment_threshold = train_df["sentiment_extreme_lag1"].quantile(0.75)

    regime_source = pd.concat([train_df, test_predictions_df[[DATE_COL]].drop_duplicates()], ignore_index=False)
    # Safer source: use the original test period feature values from test_df-equivalent rows.
    # test_predictions_df does not include feature columns, so re-load test data.
    test_df = pd.read_csv(TEST_PATH)
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])

    regime_source = test_df[[DATE_COL, "vix_lag1", "sentiment_extreme_lag1"]].drop_duplicates()

    residuals_over_time = test_predictions_df.copy()
    residuals_over_time = residuals_over_time.merge(regime_source, on=DATE_COL, how="left")

    residuals_over_time["vix_regime"] = np.where(
        residuals_over_time["vix_lag1"] >= high_vix_threshold,
        "high_vix",
        "normal_vix",
    )

    residuals_over_time["sentiment_regime"] = np.where(
        residuals_over_time["sentiment_extreme_lag1"] >= high_extreme_sentiment_threshold,
        "high_extreme_sentiment",
        "normal_extreme_sentiment",
    )

    residuals_over_time["high_vix_threshold_train_q75"] = high_vix_threshold
    residuals_over_time["high_extreme_sentiment_threshold_train_q75"] = (
        high_extreme_sentiment_threshold
    )

    regime_rows = []

    for (model_family, spec, regime_label), g in residuals_over_time.groupby(
        ["model_family", "spec", "vix_regime"]
    ):
        regime_rows.append({
            "model_family": model_family,
            "spec": spec,
            "feature_spec_role": FEATURE_SPEC_ROLES.get(spec),
            "regime_type": "vix_regime",
            "regime_label": regime_label,
            "n_obs": len(g),
            "mae": float(g["abs_error"].mean()),
            "rmse": float(np.sqrt(g["squared_error"].mean())),
            "r2": safe_r2(g["actual"].values, g["predicted"].values),
        })

    for (model_family, spec, regime_label), g in residuals_over_time.groupby(
        ["model_family", "spec", "sentiment_regime"]
    ):
        regime_rows.append({
            "model_family": model_family,
            "spec": spec,
            "feature_spec_role": FEATURE_SPEC_ROLES.get(spec),
            "regime_type": "sentiment_regime",
            "regime_label": regime_label,
            "n_obs": len(g),
            "mae": float(g["abs_error"].mean()),
            "rmse": float(np.sqrt(g["squared_error"].mean())),
            "r2": safe_r2(g["actual"].values, g["predicted"].values),
        })

    regime_error_summary = pd.DataFrame(regime_rows).sort_values(
        ["model_family", "spec", "regime_type", "regime_label"]
    ).reset_index(drop=True)

    rolling_parts = []

    for (model_family, spec), g in residuals_over_time.groupby(["model_family", "spec"]):
        g = g.sort_values(DATE_COL).copy()
        g["rolling_mae_20"] = (
            g["abs_error"]
            .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
            .mean()
        )
        g["rolling_rmse_20"] = np.sqrt(
            g["squared_error"]
            .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
            .mean()
        )
        g["rolling_mean_residual_20"] = (
            g["residual"]
            .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
            .mean()
        )
        rolling_parts.append(g)

    rolling_error_summary = pd.concat(rolling_parts, ignore_index=True)

    return residuals_over_time, regime_error_summary, rolling_error_summary


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    ensure_dirs()

    train_df, test_df = load_data()
    validate_columns(train_df)
    validate_columns(test_df)

    if not TUNED_MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing tuned master metrics file: {TUNED_MASTER_PATH}")

    if not TUNED_BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing tuned best params file: {TUNED_BEST_PARAMS_PATH}")

    tuned_master_df = pd.read_csv(TUNED_MASTER_PATH)
    best_params_df = pd.read_csv(TUNED_BEST_PARAMS_PATH)

    # ------------------------------------------------------------
    # 1. Tuned nonlinear delta table for RQ3
    # ------------------------------------------------------------
    tuned_delta_df = build_tuned_delta_vs_macro_only(tuned_master_df)

    tuned_delta_path = (
        TUNING_COMPARISON_DIR
        / "tuned_nonlinear_delta_vs_macro_only.csv"
    )
    tuned_delta_df.to_csv(tuned_delta_path, index=False)

    # ------------------------------------------------------------
    # 2. Refit all tuned nonlinear models and generate SHAP
    # ------------------------------------------------------------
    predictions_all_df, shap_values_all_df, mean_abs_shap_df = refit_predict_and_shap_all(
        train_df=train_df,
        test_df=test_df,
        best_params_df=best_params_df,
    )

    predictions_all_path = ALL_ERROR_DIR / "tuned_predictions_all.csv"
    shap_values_all_path = ALL_SHAP_DIR / "tuned_shap_values.csv"
    mean_abs_shap_path = ALL_SHAP_DIR / "tuned_mean_abs_shap_summary.csv"

    predictions_all_df.to_csv(predictions_all_path, index=False)
    shap_values_all_df.to_csv(shap_values_all_path, index=False)
    mean_abs_shap_df.to_csv(mean_abs_shap_path, index=False)

    # ------------------------------------------------------------
    # 3. Error analysis on test predictions only
    # ------------------------------------------------------------
    test_predictions_df = predictions_all_df[
        predictions_all_df["dataset"] == "test"
    ].copy()

    residuals_over_time, regime_error_summary, rolling_error_summary = (
        build_error_analysis_outputs(
            train_df=train_df,
            test_predictions_df=test_predictions_df,
        )
    )

    residuals_path = ALL_ERROR_DIR / "tuned_residuals_over_time.csv"
    regime_path = ALL_ERROR_DIR / "tuned_regime_error_summary.csv"
    rolling_path = ALL_ERROR_DIR / "tuned_rolling_error_summary.csv"

    residuals_over_time.to_csv(residuals_path, index=False)
    regime_error_summary.to_csv(regime_path, index=False)
    rolling_error_summary.to_csv(rolling_path, index=False)

    # ------------------------------------------------------------
    # 4. Manifest
    # ------------------------------------------------------------
    manifest = {
        "script": "run_day3_tuned_all_followup.py",
        "purpose": (
            "Generate full tuned nonlinear delta table, SHAP outputs, and error-analysis "
            "outputs for all tuned nonlinear feature specifications."
        ),
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "tuned_master_path": str(TUNED_MASTER_PATH),
        "tuned_best_params_path": str(TUNED_BEST_PARAMS_PATH),
        "outputs": {
            "tuned_delta_vs_macro_only": str(tuned_delta_path),
            "predictions_all": str(predictions_all_path),
            "residuals_over_time": str(residuals_path),
            "regime_error_summary": str(regime_path),
            "rolling_error_summary": str(rolling_path),
            "shap_values_all": str(shap_values_all_path),
            "mean_abs_shap_summary": str(mean_abs_shap_path),
            "individual_shap_values_dir": str(ALL_SHAP_VALUES_DIR),
            "shap_plots_dir": str(ALL_SHAP_PLOTS_DIR),
        },
        "feature_specs": FEATURE_SPECS,
        "feature_spec_roles": FEATURE_SPEC_ROLES,
        "rq3_incremental_specs": sorted(RQ3_INCREMENTAL_SPECS),
        "interpretation_note": (
            "All tuned nonlinear specifications are processed for transparency. "
            "The thesis text should report cross-model patterns and RQ-relevant findings, "
            "rather than mechanically discussing every SHAP plot or error-analysis row."
        ),
        "delta_interpretation": (
            "The tuned nonlinear delta table is restricted to RQ3 macro-sentiment specifications. "
            "RQ2 standalone sentiment specifications are excluded from the incremental delta table "
            "because they do not add sentiment to the macro-only baseline."
        ),
        "rolling_window": ROLLING_WINDOW,
        "random_state": RANDOM_STATE,
    }

    manifest_path = ALL_MODELS_DIR / "tuned_all_followup_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n=== Tuned nonlinear all-model follow-up completed successfully ===")
    print(f"[OK] Tuned nonlinear delta table: {tuned_delta_path}")
    print(f"[OK] All tuned predictions: {predictions_all_path}")
    print(f"[OK] Residuals over time: {residuals_path}")
    print(f"[OK] Regime error summary: {regime_path}")
    print(f"[OK] Rolling error summary: {rolling_path}")
    print(f"[OK] All SHAP values: {shap_values_all_path}")
    print(f"[OK] Mean absolute SHAP summary: {mean_abs_shap_path}")
    print(f"[OK] Individual SHAP values dir: {ALL_SHAP_VALUES_DIR}")
    print(f"[OK] SHAP plots dir: {ALL_SHAP_PLOTS_DIR}")
    print(f"[OK] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()