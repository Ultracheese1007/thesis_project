from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
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

TRAIN_PATH = DATA_PROCESSED_DIR / "train_daily.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test_daily.csv"
UNTUNED_MASTER_PATH = RESULTS_DIR / "comparison" / "master_metrics_table.csv"

TUNING_DIR = RESULTS_DIR / "tuning"
TUNING_RF_DIR = TUNING_DIR / "random_forest"
TUNING_XGB_DIR = TUNING_DIR / "xgboost"

DATE_COL = "date"
TARGET_COL = "nasdaq_return"
N_SPLITS = 3
RANDOM_STATE = 42


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


# Small, controlled grids
RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [5, 3, 1],  # simpler first
}

XGB_GRID = {
    "n_estimators": [100, 300],
    "learning_rate": [0.05, 0.03],  # simpler first
    "max_depth": [2, 3, 4],
}

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
        TUNING_RF_DIR / "cv_grid",
        TUNING_RF_DIR / "best_params",
        TUNING_RF_DIR / "predictions",
        TUNING_RF_DIR / "feature_importance",
        TUNING_XGB_DIR / "cv_grid",
        TUNING_XGB_DIR / "best_params",
        TUNING_XGB_DIR / "predictions",
        TUNING_XGB_DIR / "feature_importance",
        TUNING_DIR / "comparison",
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
    for cols in FEATURE_SPECS.values():
        required.update(cols)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def to_python_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


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


def build_model(model_family: str, params: dict):
    # Remove metadata fields that are not actual model hyperparameters.
    model_params = {k: v for k, v in params.items() if k != "grid_order"}

    if model_family == "random_forest":
        return RandomForestRegressor(
            **model_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_family == "xgboost":
        full_params = {**model_params, **XGB_FIXED_PARAMS}
        return XGBRegressor(**full_params)

    raise ValueError(f"Unsupported model family: {model_family}")


def generate_param_combinations(model_family: str) -> list[dict]:
    combos = []

    if model_family == "random_forest":
        grid = RF_GRID
    elif model_family == "xgboost":
        grid = XGB_GRID
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    keys = list(grid.keys())
    values_product = list(product(*(grid[k] for k in keys)))

    for grid_order, values in enumerate(values_product, start=1):
        combo = dict(zip(keys, values))
        combo["grid_order"] = grid_order
        combos.append(combo)

    return combos


# ============================================================
# 3. CV tuning
# ============================================================

def run_cv_for_param_combo(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    spec_name: str,
    params: dict,
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

        model = build_model(model_family, params)
        model.fit(X_tr_scaled, y_tr)
        val_pred = np.asarray(model.predict(X_val_scaled))

        fold_metrics = calc_metrics(y_val.values, val_pred)

        fold_rows.append({
            "model_family": model_family,
            "spec": spec_name,
            "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
            "fold": fold_id,
            "grid_order": params["grid_order"],
            "params_json": json.dumps({
                k: to_python_scalar(v)
                for k, v in params.items()
                if k != "grid_order"
            }),
            "cv_mae": fold_metrics["mae"],
            "cv_rmse": fold_metrics["rmse"],
            "cv_r2": fold_metrics["r2"],
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_start_date": train_df.iloc[val_idx][DATE_COL].min(),
            "val_end_date": train_df.iloc[val_idx][DATE_COL].max(),
        })

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "model_family": model_family,
        "spec": spec_name,
        "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
        "grid_order": params["grid_order"],
        "params_json": json.dumps({
            k: to_python_scalar(v)
            for k, v in params.items()
            if k != "grid_order"
        }),
        "cv_mean_mae": fold_df["cv_mae"].mean(),
        "cv_std_mae": fold_df["cv_mae"].std(ddof=1),
        "cv_mean_rmse": fold_df["cv_rmse"].mean(),
        "cv_std_rmse": fold_df["cv_rmse"].std(ddof=1),
        "cv_mean_r2": fold_df["cv_r2"].mean(),
        "cv_std_r2": fold_df["cv_r2"].std(ddof=1),
    }

    for k, v in params.items():
        if k != "grid_order":
            summary[k] = to_python_scalar(v)

    return fold_df, summary


def select_best_params(grid_results_df: pd.DataFrame) -> pd.Series:
    # Selection rule:
    # 1) lowest CV mean RMSE
    # 2) higher CV mean R²
    # 3) lower grid_order (simpler-first ordering)
    ranked = grid_results_df.sort_values(
        by=["cv_mean_rmse", "cv_mean_r2", "grid_order"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return ranked.iloc[0]


# ============================================================
# 4. Final tuned fit
# ============================================================

def run_final_fit_for_best_params(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    spec_name: str,
    best_params: dict,
) -> dict:
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    scaler = fit_scaler(X_train)
    X_train_scaled, X_test_scaled = transform_features(scaler, X_train, X_test)

    model = build_model(model_family, best_params)
    model.fit(X_train_scaled, y_train)

    train_pred = np.asarray(model.predict(X_train_scaled))
    test_pred = np.asarray(model.predict(X_test_scaled))

    train_metrics = calc_metrics(y_train.values, train_pred)
    test_metrics = calc_metrics(y_test.values, test_pred)

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
        "model_family": model_family,
        "spec": spec_name,
        "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
    }).sort_values("importance", ascending=False)

    if model_family == "random_forest":
        fi_dir = TUNING_RF_DIR / "feature_importance"
        pred_dir = TUNING_RF_DIR / "predictions"
    else:
        fi_dir = TUNING_XGB_DIR / "feature_importance"
        pred_dir = TUNING_XGB_DIR / "predictions"

    fi_df.to_csv(fi_dir / f"{spec_name}_tuned_feature_importance.csv", index=False)

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
    })

    train_pred_df.to_csv(pred_dir / f"{spec_name}_train_predictions_tuned.csv", index=False)
    test_pred_df.to_csv(pred_dir / f"{spec_name}_test_predictions_tuned.csv", index=False)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_pred_df": train_pred_df,
        "test_pred_df": test_pred_df,
    }


# ============================================================
# 5. Comparison tables
# ============================================================

def build_tuned_ranking_table(tuned_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = tuned_df.copy()
    ranking_df["rank_test_rmse"] = ranking_df["test_rmse"].rank(method="dense", ascending=True)
    ranking_df["rank_test_mae"] = ranking_df["test_mae"].rank(method="dense", ascending=True)
    ranking_df["rank_test_r2"] = ranking_df["test_r2"].rank(method="dense", ascending=False)

    ranking_df = ranking_df.sort_values(
        by=["rank_test_rmse", "rank_test_mae", "rank_test_r2", "model_family", "spec"]
    ).reset_index(drop=True)

    return ranking_df


def build_tuned_vs_untuned_table(
    tuned_df: pd.DataFrame,
    untuned_master_df: pd.DataFrame,
) -> pd.DataFrame:
    untuned_nonlinear = untuned_master_df[
        untuned_master_df["model_family"].isin(["random_forest", "xgboost"])
    ].copy()

    keep_cols = [
        "model_family",
        "spec",
        "cv_mean_rmse",
        "cv_mean_r2",
        "test_rmse",
        "test_r2",
    ]

    optional_cols = ["feature_spec_role"]
    for col in optional_cols:
        if col in untuned_nonlinear.columns and col not in keep_cols:
            keep_cols.insert(2, col)

    untuned_nonlinear = untuned_nonlinear[keep_cols].rename(columns={
        "cv_mean_rmse": "untuned_cv_mean_rmse",
        "cv_mean_r2": "untuned_cv_mean_r2",
        "test_rmse": "untuned_test_rmse",
        "test_r2": "untuned_test_r2",
    })

    tuned_keep_cols = [
        "model_family",
        "spec",
        "feature_spec_role",
        "cv_mean_rmse",
        "cv_mean_r2",
        "test_rmse",
        "test_r2",
    ]

    tuned_subset = tuned_df[tuned_keep_cols].rename(columns={
        "feature_spec_role": "tuned_feature_spec_role",
        "cv_mean_rmse": "tuned_cv_mean_rmse",
        "cv_mean_r2": "tuned_cv_mean_r2",
        "test_rmse": "tuned_test_rmse",
        "test_r2": "tuned_test_r2",
    })

    merged = tuned_subset.merge(
        untuned_nonlinear,
        on=["model_family", "spec"],
        how="left",
    )

    if "feature_spec_role" in merged.columns:
        merged = merged.rename(columns={
            "feature_spec_role": "untuned_feature_spec_role"
        })

    merged["delta_test_rmse_tuned_minus_untuned"] = (
        merged["tuned_test_rmse"] - merged["untuned_test_rmse"]
    )
    merged["delta_test_r2_tuned_minus_untuned"] = (
        merged["tuned_test_r2"] - merged["untuned_test_r2"]
    )

    merged["delta_error_interpretation"] = (
        "For delta_test_rmse_tuned_minus_untuned, negative values indicate that tuning lowered test RMSE. "
        "For delta_test_r2_tuned_minus_untuned, positive values indicate that tuning increased test R2."
    )

    return merged.sort_values(["model_family", "spec"]).reset_index(drop=True)


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    ensure_dirs()

    train_df, test_df = load_data()
    validate_columns(train_df)
    validate_columns(test_df)

    manifest = {
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "train_start": str(train_df[DATE_COL].min().date()),
        "train_end": str(train_df[DATE_COL].max().date()),
        "test_start": str(test_df[DATE_COL].min().date()),
        "test_end": str(test_df[DATE_COL].max().date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "cv_type": "expanding_window",
        "n_splits": N_SPLITS,
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
                    "as a standalone predictor. In nonlinear models, they are interpreted as "
                    "one-dimensional nonlinear or threshold-type signal tests, not as interaction models."
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
            "The tuning design follows the same feature-setting matrix as the fixed-parameter experiments. "
            "A full combined specification containing sentiment_mean_lag1, sentiment_std_lag1, and "
            "sentiment_extreme_lag1 together with macro predictors is not used as a primary setting, "
            "because sentiment_std_lag1 and sentiment_extreme_lag1 show high pairwise correlation and "
            "may capture overlapping sentiment-intensity information."
        ),
        "rf_grid": RF_GRID,
        "xgb_grid": XGB_GRID,
        "xgb_fixed_params": XGB_FIXED_PARAMS,
        "random_state": RANDOM_STATE,
        "selection_rule": {
            "primary": "lowest cv_mean_rmse",
            "secondary": "higher cv_mean_r2",
            "tertiary": "lower grid_order (simpler-first)",
        },
    }

    with open(TUNING_DIR / "experiment_manifest_tuning.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    tuned_rows = []
    best_params_rows = []

    for model_family in ["random_forest", "xgboost"]:
        all_param_results = []

        for spec_name, feature_cols in FEATURE_SPECS.items():
            print(f"\n=== Tuning {model_family} | {spec_name} ===")

            param_combos = generate_param_combinations(model_family)
            grid_summaries = []

            for params in param_combos:
                _, summary = run_cv_for_param_combo(
                    train_df=train_df,
                    feature_cols=feature_cols,
                    model_family=model_family,
                    spec_name=spec_name,
                    params=params,
                )
                grid_summaries.append(summary)

            grid_df = pd.DataFrame(grid_summaries).sort_values(
                by=["cv_mean_rmse", "cv_mean_r2", "grid_order"],
                ascending=[True, False, True],
            ).reset_index(drop=True)

            if model_family == "random_forest":
                grid_out = TUNING_RF_DIR / "cv_grid" / f"{spec_name}_rf_grid_results.csv"
                best_out = TUNING_RF_DIR / "best_params" / f"{spec_name}_best_params.json"
            else:
                grid_out = TUNING_XGB_DIR / "cv_grid" / f"{spec_name}_xgb_grid_results.csv"
                best_out = TUNING_XGB_DIR / "best_params" / f"{spec_name}_best_params.json"

            grid_df.to_csv(grid_out, index=False)

            best_row = select_best_params(grid_df)
            best_params = {
                k: to_python_scalar(best_row[k])
                for k in best_row.index
                if k in ["n_estimators", "max_depth", "min_samples_leaf", "learning_rate"]
            }

            with open(best_out, "w", encoding="utf-8") as f:
                json.dump({
                    "model_family": model_family,
                    "spec": spec_name,
                    "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                    "best_params": best_params,
                    "cv_mean_rmse": float(best_row["cv_mean_rmse"]),
                    "cv_mean_r2": float(best_row["cv_mean_r2"]),
                }, f, indent=2)

            final_result = run_final_fit_for_best_params(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                model_family=model_family,
                spec_name=spec_name,
                best_params=best_params,
            )

            tuned_rows.append({
                "model_family": model_family,
                "spec": spec_name,
                "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                "n_features": len(feature_cols),
                "feature_list": ", ".join(feature_cols),

                "cv_mean_mae": float(best_row["cv_mean_mae"]),
                "cv_std_mae": float(best_row["cv_std_mae"]),
                "cv_mean_rmse": float(best_row["cv_mean_rmse"]),
                "cv_std_rmse": float(best_row["cv_std_rmse"]),
                "cv_mean_r2": float(best_row["cv_mean_r2"]),
                "cv_std_r2": float(best_row["cv_std_r2"]),

                "train_mae": final_result["train_metrics"]["mae"],
                "train_rmse": final_result["train_metrics"]["rmse"],
                "train_r2": final_result["train_metrics"]["r2"],

                "test_mae": final_result["test_metrics"]["mae"],
                "test_rmse": final_result["test_metrics"]["rmse"],
                "test_r2": final_result["test_metrics"]["r2"],
            })

            best_params_rows.append({
                "model_family": model_family,
                "spec": spec_name,
                "feature_spec_role": FEATURE_SPEC_ROLES.get(spec_name),
                **best_params,
                "cv_mean_rmse": float(best_row["cv_mean_rmse"]),
                "cv_mean_r2": float(best_row["cv_mean_r2"]),
            })

            all_param_results.append(grid_df)

        all_param_results_df = pd.concat(all_param_results, ignore_index=True)

        if model_family == "random_forest":
            all_param_results_df.to_csv(
                TUNING_RF_DIR / "random_forest_all_grid_results.csv",
                index=False,
            )
        else:
            all_param_results_df.to_csv(
                TUNING_XGB_DIR / "xgboost_all_grid_results.csv",
                index=False,
            )

    tuned_master_df = pd.DataFrame(tuned_rows).sort_values(
        ["model_family", "spec"]
    ).reset_index(drop=True)

    tuned_best_params_df = pd.DataFrame(best_params_rows).sort_values(
        ["model_family", "spec"]
    ).reset_index(drop=True)

    tuned_master_path = TUNING_DIR / "comparison" / "tuned_nonlinear_master_metrics.csv"
    tuned_best_params_path = TUNING_DIR / "comparison" / "tuned_best_params_summary.csv"
    tuned_ranking_path = TUNING_DIR / "comparison" / "tuned_nonlinear_ranking_table.csv"

    tuned_master_df.to_csv(tuned_master_path, index=False)
    tuned_best_params_df.to_csv(tuned_best_params_path, index=False)

    tuned_ranking_df = build_tuned_ranking_table(tuned_master_df)
    tuned_ranking_df.to_csv(tuned_ranking_path, index=False)

    if UNTUNED_MASTER_PATH.exists():
        untuned_master_df = pd.read_csv(UNTUNED_MASTER_PATH)
        compare_df = build_tuned_vs_untuned_table(tuned_master_df, untuned_master_df)
        compare_path = TUNING_DIR / "comparison" / "tuned_vs_untuned_nonlinear.csv"
        compare_df.to_csv(compare_path, index=False)
        print(f"[OK] Saved tuned vs untuned comparison: {compare_path}")
    else:
        print(f"[WARN] Untuned master table not found: {UNTUNED_MASTER_PATH}")
        print("[WARN] tuned_vs_untuned_nonlinear.csv was not created.")

    print("\n=== Nonlinear tuning completed successfully ===")
    print(f"[OK] Tuned nonlinear master metrics: {tuned_master_path}")
    print(f"[OK] Tuned best params summary: {tuned_best_params_path}")
    print(f"[OK] Tuned nonlinear ranking table: {tuned_ranking_path}")


if __name__ == "__main__":
    main()