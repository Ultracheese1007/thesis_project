from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

#build final data set,split train set and test set.
#EDA check

# =========================
# 0. Paths
# =========================
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

RESULTS_DIR = PROJECT_ROOT / "results"
SANITY_DIR = RESULTS_DIR / "sanity_checks"
DESC_DIR = RESULTS_DIR / "descriptive"
EDA_FIG_DIR = DESC_DIR / "eda_figures"


# =========================
# 1. Directory setup
# =========================
def ensure_directories() -> None:
    required_dirs = [
        INTERIM_DIR,
        PROCESSED_DIR,
        SANITY_DIR,
        DESC_DIR,
        EDA_FIG_DIR,
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Utilities
# =========================
def standardize_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def save_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


# =========================
# 3. Load and clean source tables
# =========================
def load_nasdaq() -> pd.DataFrame:
    """
    Strict Day 2 assumption:
    nasdaq100_close.csv must contain exactly:
    - Date
    - nasdaq_close
    """
    path = RAW_DIR / "nasdaq100_close.csv"
    df = pd.read_csv(path)

    required_cols = {"Date", "nasdaq_close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[nasdaq100_close.csv] Missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = standardize_date_column(df, "Date")
    df = df.rename(columns={"Date": "date"})
    df["nasdaq_close"] = coerce_numeric(df["nasdaq_close"])

    # Drop rows without valid close
    df = df.dropna(subset=["nasdaq_close"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Daily log return on trading-day panel
    df["nasdaq_return"] = np.log(df["nasdaq_close"] / df["nasdaq_close"].shift(1))

    return df[["date", "nasdaq_close", "nasdaq_return"]].copy()


def load_dgs10() -> pd.DataFrame:
    """
    Strict Day 2 assumption:
    DGS10.csv must contain exactly:
    - observation_date
    - DGS10
    """
    path = RAW_DIR / "DGS10.csv"
    df = pd.read_csv(path)

    required_cols = {"observation_date", "DGS10"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[DGS10.csv] Missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = standardize_date_column(df, "observation_date")
    df = df.rename(columns={"observation_date": "date", "DGS10": "dgs10"})
    df["dgs10"] = coerce_numeric(df["dgs10"])

    return df[["date", "dgs10"]].copy()


def load_vix() -> pd.DataFrame:
    """
    Strict Day 2 assumption:
    VIXCLS.csv must contain exactly:
    - observation_date
    - VIXCLS
    """
    path = RAW_DIR / "VIXCLS.csv"
    df = pd.read_csv(path)

    required_cols = {"observation_date", "VIXCLS"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[VIXCLS.csv] Missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = standardize_date_column(df, "observation_date")
    df = df.rename(columns={"observation_date": "date", "VIXCLS": "vix"})
    df["vix"] = coerce_numeric(df["vix"])

    return df[["date", "vix"]].copy()


def load_sentiment() -> pd.DataFrame:
    path = PROCESSED_DIR / "daily_sentiment.csv"
    df = pd.read_csv(path)

    required_cols = {"date", "sentiment_mean", "sentiment_std", "sentiment_extreme"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[daily_sentiment.csv] Missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = standardize_date_column(df, "date")
    for c in ["sentiment_mean", "sentiment_std", "sentiment_extreme"]:
        df[c] = coerce_numeric(df[c])

    return df[["date", "sentiment_mean", "sentiment_std", "sentiment_extreme"]].copy()


# =========================
# 4. Merge logic (strict Day 2 design)
# =========================
def build_merged_daily_base(
    nasdaq: pd.DataFrame,
    dgs10: pd.DataFrame,
    vix: pd.DataFrame,
    sentiment: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        merged_base: trading-day master table after alignment / fill rules
        alignment_checks: row-level summary table for merge diagnostics
    """
    base = nasdaq.copy()
    base["master_calendar"] = "nasdaq_trading_day"

    n_before_merge = len(base)

    # Exact left merge macro onto trading-day master panel
    base = base.merge(dgs10, on="date", how="left")
    base = base.merge(vix, on="date", how="left")

    # Forward-fill macro after alignment
    base["dgs10"] = base["dgs10"].ffill()
    base["vix"] = base["vix"].ffill()

    # Exact left merge sentiment onto trading-day panel
    base = base.merge(sentiment, on="date", how="left")

    # Missing sentiment on trading days -> 0
    sentiment_cols = ["sentiment_mean", "sentiment_std", "sentiment_extreme"]
    base[sentiment_cols] = base[sentiment_cols].fillna(0.0)

    n_after_merge = len(base)

    alignment_checks = pd.DataFrame(
        [
            {
                "component": "nasdaq_master_calendar",
                "row_count": len(nasdaq),
                "date_min": nasdaq["date"].min(),
                "date_max": nasdaq["date"].max(),
                "note": "Master calendar based on trading days with valid NASDAQ close."
            },
            {
                "component": "dgs10_source",
                "row_count": len(dgs10),
                "date_min": dgs10["date"].min(),
                "date_max": dgs10["date"].max(),
                "note": "Merged by exact date onto master calendar, then forward-filled."
            },
            {
                "component": "vix_source",
                "row_count": len(vix),
                "date_min": vix["date"].min(),
                "date_max": vix["date"].max(),
                "note": "Merged by exact date onto master calendar, then forward-filled."
            },
            {
                "component": "sentiment_source",
                "row_count": len(sentiment),
                "date_min": sentiment["date"].min(),
                "date_max": sentiment["date"].max(),
                "note": "Merged by exact date onto trading-day panel; missing trading-day sentiment filled with 0."
            },
            {
                "component": "merged_daily_base",
                "row_count": n_after_merge,
                "date_min": base["date"].min(),
                "date_max": base["date"].max(),
                "note": f"Trading-day master panel preserved. Rows before merge = {n_before_merge}, after merge = {n_after_merge}."
            }
        ]
    )

    return base, alignment_checks


# =========================
# 5. Lag creation and final dataset
# =========================
def build_final_daily_dataset(merged_base: pd.DataFrame) -> pd.DataFrame:
    final_df = merged_base.copy()

    lag_source_cols = [
        "dgs10",
        "vix",
        "sentiment_mean",
        "sentiment_std",
        "sentiment_extreme",
    ]

    for col in lag_source_cols:
        final_df[f"{col}_lag1"] = final_df[col].shift(1)

    required_final_cols = [
        "nasdaq_return",
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_std_lag1",
        "sentiment_extreme_lag1",
    ]
    final_df = final_df.dropna(subset=required_final_cols).copy()
    final_df = final_df.sort_values("date").reset_index(drop=True)

    return final_df


# =========================
# 6. Split
# =========================
def split_train_test(final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = final_df[
        (final_df["date"] >= pd.Timestamp("2020-01-01")) &
        (final_df["date"] <= pd.Timestamp("2023-12-31"))
    ].copy()

    test_df = final_df[
        (final_df["date"] >= pd.Timestamp("2024-01-01")) &
        (final_df["date"] <= pd.Timestamp("2024-12-31"))
    ].copy()

    return train_df, test_df


# =========================
# 7. Sanity checks
# =========================
def build_missingness_summary(
    merged_base: pd.DataFrame,
    final_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    def add_missing_block(df: pd.DataFrame, stage_name: str) -> None:
        for col in df.columns:
            rows.append(
                {
                    "stage": stage_name,
                    "column": col,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_ratio": float(df[col].isna().mean()),
                    "row_count": int(len(df)),
                }
            )

    add_missing_block(merged_base, "merged_daily_base")
    add_missing_block(final_df, "final_daily_dataset")
    add_missing_block(train_df, "train_daily")
    add_missing_block(test_df, "test_daily")

    return pd.DataFrame(rows)


def build_leakage_checks_md() -> str:
    return """# Day 2 Leakage and Alignment Checks

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
"""


# =========================
# 8. Descriptive stats and correlations
# =========================
def build_summary_statistics(final_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "nasdaq_return",
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_std_lag1",
        "sentiment_extreme_lag1",
    ]
    summary = final_df[cols].describe().T.reset_index().rename(columns={"index": "variable"})
    return summary


def build_correlation_matrix(final_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "nasdaq_return",
        "dgs10_lag1",
        "vix_lag1",
        "sentiment_mean_lag1",
        "sentiment_std_lag1",
        "sentiment_extreme_lag1",
    ]
    return final_df[cols].corr()


# =========================
# 9. EDA figures
# =========================
def plot_single_timeseries(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path
) -> None:
    plt.figure(figsize=(11, 4))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_two_series(
    df: pd.DataFrame,
    x_col: str,
    y1: str,
    y2: str,
    title: str,
    output_path: Path
) -> None:
    plt.figure(figsize=(11, 4))
    plt.plot(df[x_col], df[y1], label=y1)
    plt.plot(df[x_col], df[y2], label=y2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_three_series(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    output_path: Path
) -> None:
    plt.figure(figsize=(11, 4))
    for y in y_cols:
        plt.plot(df[x_col], df[y], label=y)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(corr_df.values, aspect="auto")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(cax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_eda_figures(merged_base: pd.DataFrame, final_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    plot_single_timeseries(
        final_df, "date", "nasdaq_return",
        "NASDAQ-100 Daily Log Return Over Time",
        "nasdaq_return",
        EDA_FIG_DIR / "nasdaq_return_over_time.png"
    )

    plot_two_series(
        merged_base, "date", "dgs10", "vix",
        "Macro Series Over Time",
        EDA_FIG_DIR / "macro_series_over_time.png"
    )

    plot_three_series(
        merged_base, "date",
        ["sentiment_mean", "sentiment_std", "sentiment_extreme"],
        "Sentiment Series Over Time",
        EDA_FIG_DIR / "sentiment_series_over_time.png"
    )

    plot_correlation_heatmap(
        corr_df,
        EDA_FIG_DIR / "correlation_heatmap.png"
    )

    plot_single_timeseries(
        merged_base, "date", "vix",
        "VIX Over Time",
        "vix",
        EDA_FIG_DIR / "vix_over_time.png"
    )

    plot_single_timeseries(
        merged_base, "date", "sentiment_extreme",
        "Extreme Sentiment Over Time",
        "sentiment_extreme",
        EDA_FIG_DIR / "sentiment_extreme_over_time.png"
    )


# =========================
# 10. Main
# =========================
def main() -> None:
    ensure_directories()

    # Load source tables
    nasdaq = load_nasdaq()
    dgs10 = load_dgs10()
    vix = load_vix()
    sentiment = load_sentiment()

    # Build merged base
    merged_base, alignment_checks = build_merged_daily_base(nasdaq, dgs10, vix, sentiment)

    # Build final dataset with lagged predictors
    final_df = build_final_daily_dataset(merged_base)

    # Split
    train_df, test_df = split_train_test(final_df)

    # Sanity checks
    missingness_df = build_missingness_summary(merged_base, final_df, train_df, test_df)
    leakage_md = build_leakage_checks_md()

    # Descriptive
    summary_stats_df = build_summary_statistics(final_df)
    corr_df = build_correlation_matrix(final_df)

    # Figures
    create_eda_figures(merged_base, final_df, corr_df)

    # Save outputs
    save_dataframe(merged_base, INTERIM_DIR / "merged_daily_base.csv")
    save_dataframe(final_df, PROCESSED_DIR / "final_daily_dataset.csv")
    save_dataframe(train_df, PROCESSED_DIR / "train_daily.csv")
    save_dataframe(test_df, PROCESSED_DIR / "test_daily.csv")

    save_dataframe(missingness_df, SANITY_DIR / "missingness_summary.csv")
    save_dataframe(alignment_checks, SANITY_DIR / "alignment_checks.csv")
    save_markdown(SANITY_DIR / "leakage_checks.md", leakage_md)

    summary_stats_df.to_csv(DESC_DIR / "summary_statistics.csv", index=False)
    corr_df.to_csv(DESC_DIR / "correlation_matrix.csv", index=True)

    print("Day 2 dataset build completed successfully.")
    print("Generated outputs:")
    print(" - data/interim/merged_daily_base.csv")
    print(" - data/processed/final_daily_dataset.csv")
    print(" - data/processed/train_daily.csv")
    print(" - data/processed/test_daily.csv")
    print(" - results/sanity_checks/missingness_summary.csv")
    print(" - results/sanity_checks/alignment_checks.csv")
    print(" - results/sanity_checks/leakage_checks.md")
    print(" - results/descriptive/summary_statistics.csv")
    print(" - results/descriptive/correlation_matrix.csv")
    print(" - results/descriptive/eda_figures/")


if __name__ == "__main__":
    main()