"""
Generate Figure 4.1: Predicted vs Actual NASDAQ-100 Daily Returns (2024).

This script reads test-set predictions for the macro_mean_extreme specification
from three selected models (OLS, tuned Random Forest, tuned XGBoost) and
overlays them on the actual NASDAQ-100 daily log returns for the 2024
hold-out test period.

Output:
results/figures/figure_4_1_predicted_vs_actual_2024.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ----- Paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

OLS_PRED_FILE = (
    RESULTS_DIR
    / "models"
    / "ols"
    / "predictions"
    / "macro_mean_extreme_test_predictions.csv"
)

RF_PRED_FILE = (
    RESULTS_DIR
    / "tuning"
    / "random_forest"
    / "predictions"
    / "macro_mean_extreme_test_predictions_tuned.csv"
)

XGB_PRED_FILE = (
    RESULTS_DIR
    / "tuning"
    / "xgboost"
    / "predictions"
    / "macro_mean_extreme_test_predictions_tuned.csv"
)

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = FIG_DIR / "figure_4_1_predicted_vs_actual_2024.png"


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def validate_prediction_columns(df: pd.DataFrame, file_path: Path) -> None:
    required_cols = {"date", "actual", "predicted"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{file_path} is missing required columns: {sorted(missing_cols)}"
        )


def main() -> None:
    # ----- Validate input files -----
    for path in [OLS_PRED_FILE, RF_PRED_FILE, XGB_PRED_FILE]:
        validate_input_file(path)

    # ----- Load data -----
    ols = pd.read_csv(OLS_PRED_FILE, parse_dates=["date"])
    rf = pd.read_csv(RF_PRED_FILE, parse_dates=["date"])
    xgb = pd.read_csv(XGB_PRED_FILE, parse_dates=["date"])

    for df_, path in [
        (ols, OLS_PRED_FILE),
        (rf, RF_PRED_FILE),
        (xgb, XGB_PRED_FILE),
    ]:
        validate_prediction_columns(df_, path)

    # ----- Build aligned dataframe -----
    df = ols[["date", "actual"]].copy()
    df["ols_pred"] = ols["predicted"].values
    df["rf_pred"] = rf["predicted"].values
    df["xgb_pred"] = xgb["predicted"].values

    # Convert daily log returns to percentage for display
    df["actual_pct"] = df["actual"] * 100
    df["ols_pct"] = df["ols_pred"] * 100
    df["rf_pct"] = df["rf_pred"] * 100
    df["xgb_pct"] = df["xgb_pred"] * 100

    # ----- Plot: thesis-ready black-and-white figure -----
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)

    ax.plot(
        df["date"],
        df["actual_pct"],
        color="black",
        linewidth=1.2,
        linestyle="-",
        label="Actual",
    )

    ax.plot(
        df["date"],
        df["ols_pct"],
        color="#666666",
        linewidth=1.0,
        linestyle=(0, (6, 3)),
        label="OLS macro_mean_extreme",
    )

    ax.plot(
        df["date"],
        df["rf_pct"],
        color="#444444",
        linewidth=1.0,
        linestyle=(0, (2, 2)),
        label="Tuned RF macro_mean_extreme",
    )

    ax.plot(
        df["date"],
        df["xgb_pct"],
        color="#222222",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label="Tuned XGBoost macro_mean_extreme",
    )

    # Zero reference line
    ax.axhline(
        0,
        color="black",
        linewidth=0.4,
        linestyle=":",
        alpha=0.5,
    )

    # Axes
    ax.set_xlabel("Trading day (2024)", fontsize=11)
    ax.set_ylabel("Daily log return (%)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    # Monthly x-axis ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Grid
    ax.grid(True, linewidth=0.4, alpha=0.3, color="black")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    # Y-axis range
    ax.set_ylim(-4, 3.5)

    # Save
    plt.tight_layout()
    plt.savefig(OUT_FILE, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()

    print(f"Figure 4.1 saved to: {OUT_FILE}")
    print(f"  N test observations: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(
        f"  Actual range: {df['actual_pct'].min():.2f}% "
        f"to {df['actual_pct'].max():.2f}%"
    )


if __name__ == "__main__":
    main()
