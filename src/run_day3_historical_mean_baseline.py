from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# 0. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"
RESULTS_COMPARISON_DIR = RESULTS_DIR / "comparison"

TRAIN_PATH = DATA_PROCESSED_DIR / "train_daily.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test_daily.csv"
MASTER_METRICS_PATH = RESULTS_COMPARISON_DIR / "master_metrics_table.csv"

DATE_COL = "date"
TARGET_COL = "nasdaq_return"
N_SPLITS = 3


# ============================================================
# 1. Utilities
# ============================================================

def ensure_dirs() -> None:
    dirs = [
        RESULTS_MODELS_DIR / "historical_mean" / "cv",
        RESULTS_MODELS_DIR / "historical_mean" / "predictions",
        RESULTS_COMPARISON_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in (train_df, test_df):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    required_cols = {DATE_COL, TARGET_COL}
    for df_name, df in [("train_df", train_df), ("test_df", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{df_name} missing required columns: {sorted(missing)}")

    return train_df, test_df


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


def predict_historical_mean(train_target: pd.Series, n_obs: int) -> np.ndarray:
    mean_value = float(train_target.mean())
    return np.full(shape=n_obs, fill_value=mean_value, dtype=float)


# ============================================================
# 2. CV
# ============================================================

def run_cv_historical_mean(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y_all = train_df[TARGET_COL].copy()
    tscv = make_tscv()

    fold_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(tscv.split(y_all), start=1):
        y_tr = y_all.iloc[train_idx].copy()
        y_val = y_all.iloc[val_idx].copy()

        val_pred = predict_historical_mean(y_tr, len(y_val))
        fold_metrics = calc_metrics(y_val.values, val_pred)

        fold_rows.append({
            "model_family": "historical_mean",
            "spec": "historical_mean",
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
# 3. Final train/test predictions
# ============================================================

def run_final_historical_mean(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    y_train = train_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()

    train_pred = predict_historical_mean(y_train, len(y_train))
    test_pred = predict_historical_mean(y_train, len(y_test))

    train_metrics = calc_metrics(y_train.values, train_pred)
    test_metrics = calc_metrics(y_test.values, test_pred)

    train_pred_df = pd.DataFrame({
        DATE_COL: train_df[DATE_COL],
        "actual": y_train.values,
        "predicted": train_pred,
        "residual": y_train.values - train_pred,
        "abs_error": np.abs(y_train.values - train_pred),
        "squared_error": (y_train.values - train_pred) ** 2,
        "model_family": "historical_mean",
        "spec": "historical_mean",
        "dataset": "train",
    })

    test_pred_df = pd.DataFrame({
        DATE_COL: test_df[DATE_COL],
        "actual": y_test.values,
        "predicted": test_pred,
        "residual": y_test.values - test_pred,
        "abs_error": np.abs(y_test.values - test_pred),
        "squared_error": (y_test.values - test_pred) ** 2,
        "model_family": "historical_mean",
        "spec": "historical_mean",
        "dataset": "test",
    })

    pred_dir = RESULTS_MODELS_DIR / "historical_mean" / "predictions"
    train_pred_df.to_csv(pred_dir / "historical_mean_train_predictions.csv", index=False)
    test_pred_df.to_csv(pred_dir / "historical_mean_test_predictions.csv", index=False)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_pred_df": train_pred_df,
        "test_pred_df": test_pred_df,
    }


# ============================================================
# 4. Ranking helper
# ============================================================

def build_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = df.copy()
    ranking_df["rank_test_rmse"] = ranking_df["test_rmse"].rank(method="dense", ascending=True)
    ranking_df["rank_test_mae"] = ranking_df["test_mae"].rank(method="dense", ascending=True)
    ranking_df["rank_test_r2"] = ranking_df["test_r2"].rank(method="dense", ascending=False)

    ranking_df = ranking_df.sort_values(
        by=["rank_test_rmse", "rank_test_mae", "rank_test_r2", "model_family", "spec"]
    ).reset_index(drop=True)

    return ranking_df


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    ensure_dirs()
    train_df, test_df = load_data()

    # -------------------------
    # CV
    # -------------------------
    cv_fold_df, cv_summary = run_cv_historical_mean(train_df)
    cv_out = RESULTS_MODELS_DIR / "historical_mean" / "cv" / "historical_mean_cv_fold_metrics.csv"
    cv_fold_df.to_csv(cv_out, index=False)

    # -------------------------
    # Final train/test
    # -------------------------
    final_result = run_final_historical_mean(train_df, test_df)

    # -------------------------
    # Summary table for historical mean baseline
    # -------------------------
    hm_row = {
        "model_family": "historical_mean",
        "spec": "historical_mean",
        "feature_spec_role": "naive_baseline",
        "n_features": 0,
        "feature_list": "<none>",
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

    hm_df = pd.DataFrame([hm_row])
    hm_path = RESULTS_COMPARISON_DIR / "historical_mean_metrics_table.csv"
    hm_df.to_csv(hm_path, index=False)

    # -------------------------
    # Merge with existing master table
    # -------------------------
    if MASTER_METRICS_PATH.exists():
        master_df = pd.read_csv(MASTER_METRICS_PATH)
        combined_df = pd.concat([master_df, hm_df], ignore_index=True)
        combined_df = combined_df.sort_values(["model_family", "spec"]).reset_index(drop=True)

        combined_path = RESULTS_COMPARISON_DIR / "master_metrics_table_with_historical_mean.csv"
        combined_df.to_csv(combined_path, index=False)

        ranking_df = build_ranking_table(combined_df)
        ranking_path = RESULTS_COMPARISON_DIR / "model_ranking_table_with_historical_mean.csv"
        ranking_df.to_csv(ranking_path, index=False)

        print(f"[OK] Saved merged master table: {combined_path}")
        print(f"[OK] Saved merged ranking table: {ranking_path}")
    else:
        print("[WARN] master_metrics_table.csv not found. Only historical_mean_metrics_table.csv was created.")

    print(f"[OK] Saved CV fold metrics: {cv_out}")
    print(f"[OK] Saved historical mean metrics: {hm_path}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()
