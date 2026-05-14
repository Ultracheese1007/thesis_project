# NASDAQ-100 Daily Return Prediction with Macro Indicators and FinBERT News Sentiment

Master thesis empirical pipeline. The project evaluates whether daily institutional-news sentiment (extracted with FinBERT) adds predictive information for NASDAQ-100 daily log returns beyond a macro-financial baseline. The full pipeline runs end-to-end from raw data download through model comparison, hyperparameter tuning, SHAP interpretation, and error analysis.

---

## 1. Research questions

The empirical design is structured around three research questions, each mapped to a fixed set of feature specifications (see `docs/model_info_sheet.md` for the authoritative spec).

- **RQ1 — Macro baseline.** To what extent can macroeconomic indicators (DGS10, VIX) explain or predict the daily returns of AI technology stocks? Do predictive performances differ between linear (OLS) and nonlinear (XGBoost) models?
- **RQ2 — Standalone sentiment.** To what extent can the institutional sentiment index constructed using FinBERT predict the daily returns of AI technology stocks? How does it perform as a single feature in OLS and XGBoost models?
- **RQ3 — Incremental value.** When macroeconomic indicators and institutional sentiment features are simultaneously incorporated into the predictive model, does its predictive performance significantly improve relative to single-feature models (e.g., via ΔR² or ΔMAE)? Within the SHAP framework, how do the relative contributions of macroeconomic and sentiment features differ?

The AI technology stock narrative motivates the research context. The formal empirical target is `nasdaq_return`, the daily log return of the NASDAQ-100 index, which serves as a liquid, observable proxy for the AI-tech-heavy large-cap segment. All predictors are lagged one trading day to avoid look-ahead bias.

---

## 2. Data

| Source | File | Coverage | Notes |
|---|---|---|---|
| Yahoo Finance (`^NDX`) | `data/raw/nasdaq100_close.csv` | 2020-01-01 → 2024-12-31 | Daily close, basis for `nasdaq_return` |
| FRED `DGS10` | `data/raw/DGS10.csv` | same | 10-year Treasury yield |
| FRED `VIXCLS` | `data/raw/VIXCLS.csv` | same | CBOE Volatility Index |
| GDELT 15-minute event stream | (not bundled — produced by `gdelt_collect_daily.py`) | 2020-01-01 → 2024-12-31 | Institutional outlets only: Reuters, Bloomberg, CNBC, FT, Yahoo Finance, MarketWatch |
| FinBERT (`ProsusAI/finbert`) | `data/processed/daily_sentiment.csv` | same | Daily aggregates: `sentiment_mean`, `sentiment_std`, `sentiment_extreme` |

The raw news corpus is not bundled in the repository because of its size. The aggregated daily sentiment file (`daily_sentiment.csv`) is provided so the modeling pipeline can be reproduced without rerunning FinBERT inference.

The frozen modeling dataset is `data/processed/final_daily_dataset.csv` (1257 trading days). It is split chronologically into `train_daily.csv` (2020-01-03 → 2023-12-29, n = 1005) and `test_daily.csv` (2024-01-02 → 2024-12-31, n = 252). The 2024 hold-out is never touched during model selection or tuning.

Feature definitions, RQ role mapping, and column-level provenance are stored in `data/processed/feature_manifest.json`, and the split parameters in `data/processed/split_manifest.json`.

---

## 3. Project structure

```
thesis_project/
├── README.md                              ← this file
├── requirements.txt
├── docs/
│   └── model_info_sheet.md                ← authoritative spec: target, features, CV, metrics
├── data/
│   ├── raw/                               ← NDX close, DGS10, VIXCLS
│   ├── interim/                           ← merged_daily_base.csv (post-merge, pre-lag)
│   └── processed/                         ← final_daily_dataset, train/test, manifests
├── src/
│   ├── nasdaq100_download.py              ← Step 1: download NASDAQ-100 close
│   ├── gdelt_collect_daily.py             ← Step 2: collect institutional news from GDELT
│   ├── finbert_daily_aggregate.py         ← Step 3: FinBERT inference → daily aggregates
│   ├── build_final_daily_dataset_day2.py  ← Step 4: merge, lag, split, EDA, sanity checks
│   ├── setup_specification.py             ← Step 5: persist feature manifest
│   ├── run_day3_core_experiments.py       ← Step 6: 27 untuned experiments + SHAP
│   ├── run_day3_historical_mean_baseline.py ← Step 7: naive baseline
│   ├── run_day3_nonlinear_tuning.py       ← Step 8: grid-search tuning for RF and XGBoost
│   ├── run_day3_tuned_all_followup.py     ← Step 9: rerun with tuned params, SHAP, errors
│   └── build_figure_4_1.py                ← Step 10: Figure 4.1 (predicted vs actual 2024)
├── results/
│   ├── sanity_checks/                     ← leakage_checks.md, alignment, missingness
│   ├── descriptive/                       ← summary stats, correlations, EDA figures
│   ├── models/                            ← per-model predictions, CV folds, importances
│   │   ├── ols/        random_forest/     xgboost/        historical_mean/
│   ├── tuning/                            ← grid results, best params, tuned predictions
│   │   ├── random_forest/  xgboost/  all_models/  comparison/
│   ├── comparison/                        ← master metrics tables, ranking, delta tables
│   ├── shap/                              ← SHAP summaries (untuned reference run)
│   ├── error_analysis/                    ← residuals, rolling errors, regime errors
│   └── figures/                           ← thesis figures (Figure 4.1 etc.)
└── thesis/                                ← (manuscript notes, kept separate from code)
```

---

## 4. Environment

Python 3.11+ is recommended. The pipeline was developed and frozen with the package versions in `requirements.txt`.

```bash
python -m venv thesis_daily_env
source thesis_daily_env/bin/activate          # Windows: thesis_daily_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: `finbert_daily_aggregate.py` additionally requires `torch` and `transformers`. Install whichever PyTorch build matches your platform (CPU is sufficient — FinBERT inference takes ~1–2 hours on the full 2020–2024 news corpus on CPU; a GPU brings it under 15 minutes):

```bash
pip install torch transformers
```

`gdelt_collect_daily.py` requires `trafilatura` for article body extraction:

```bash
pip install trafilatura
```

---

## 5. Reproducing the experiments

All scripts are designed to be run from the project root so that relative paths resolve correctly:

```bash
cd thesis_project
```

### 5.1 Full pipeline (from raw data)

| # | Script | What it produces | Approx. runtime |
|---|---|---|---|
| 1 | `python src/nasdaq100_download.py` | `data/raw/nasdaq100_close.csv` | seconds |
| 2 | `python src/gdelt_collect_daily.py` | per-day institutional news CSVs | hours (network-bound) |
| 3 | `python src/finbert_daily_aggregate.py` | `data/processed/daily_sentiment.csv` | 1–2 h CPU / ~15 min GPU |
| 4 | `python src/build_final_daily_dataset_day2.py` | `data/processed/final_daily_dataset.csv`, train/test split, EDA figures, sanity checks | < 1 min |
| 5 | `python src/setup_specification.py` | `feature_manifest.json` | seconds |
| 6 | `python src/run_day3_core_experiments.py` | 27 untuned model runs (OLS + RF + XGB × 9 specs), CV metrics, predictions, untuned SHAP | ~10–30 min |
| 7 | `python src/run_day3_historical_mean_baseline.py` | naive baseline, appended to master metrics table | < 1 min |
| 8 | `python src/run_day3_nonlinear_tuning.py` | grid search for RF and XGBoost, best params per spec | 30–60 min |
| 9 | `python src/run_day3_tuned_all_followup.py` | tuned reruns + tuned SHAP + error analysis | 10–20 min |
| 10 | `python src/build_figure_4_1.py` | `results/figures/figure_4_1_predicted_vs_actual_2024.png` | seconds |

DGS10 and VIXCLS were downloaded manually from FRED and placed in `data/raw/`. To refresh them, get the CSVs from https://fred.stlouisfed.org/series/DGS10 and https://fred.stlouisfed.org/series/VIXCLS.

### 5.2 Modeling only (skip data collection)

Because the aggregated daily sentiment and the frozen modeling dataset are already in `data/processed/`, you can reproduce the modeling stage and all results tables without rerunning Steps 1–3:

```bash
python src/build_final_daily_dataset_day2.py
python src/setup_specification.py
python src/run_day3_core_experiments.py
python src/run_day3_historical_mean_baseline.py
python src/run_day3_nonlinear_tuning.py
python src/run_day3_tuned_all_followup.py
python src/build_figure_4_1.py
```

---

## 6. Methodology summary

- **Target:** `nasdaq_return` = ln(close_t / close_{t-1}).
- **Predictors:** all lagged by one trading day. Macro family — `dgs10_lag1`, `vix_lag1`. Sentiment family — `sentiment_mean_lag1`, `sentiment_std_lag1`, `sentiment_extreme_lag1`.
- **Feature specifications (9 total):**
  - RQ1: `macro_only`
  - RQ2: `sentiment_mean_only`, `sentiment_std_only`, `sentiment_extreme_only`
  - RQ3: `macro_mean`, `macro_std`, `macro_extreme`, `macro_mean_std`, `macro_mean_extreme`
- **Models:** OLS, Random Forest, XGBoost. A historical-mean predictor is included as a naive baseline.
- **Split:** chronological only. Train 2020-01-03 → 2023-12-29 (n = 1005). Test 2024-01-02 → 2024-12-31 (n = 252).
- **Cross-validation:** expanding-window `TimeSeriesSplit` with `n_splits = 3`, fit on training data only.
- **Standardization:** fit on the training fold only, applied to the validation/test fold. Never fit on the full dataset.
- **Hyperparameter selection:** lowest mean CV RMSE, tie-broken by higher mean CV R² and then by lower `grid_order`. The 2024 hold-out is never used for selection.
- **Metrics:** MAE, RMSE, R² on CV / train / test. For RQ3, delta metrics (ΔMAE, ΔRMSE, ΔR²) are computed against the `macro_only` reference within the same model family, quantifying the incremental value of adding sentiment features on top of the macro baseline.
- **SHAP:** computed for tuned nonlinear models on the test set. Read as model-attribution, not causal effect.

Leakage controls are documented in `results/sanity_checks/leakage_checks.md` and are encoded directly into the pipeline (lag-after-align, fit-scaler-on-train-only, no random shuffling anywhere).

---

## 7. Where to find the main results

| Thesis claim | File |
|---|---|
| Master metrics across all 27 settings | `results/comparison/master_metrics_table.csv` |
| Same, with historical-mean baseline appended | `results/comparison/master_metrics_table_with_historical_mean.csv` |
| RQ3 delta metrics vs `macro_only` | `results/comparison/delta_metrics_table.csv` |
| Model ranking | `results/comparison/model_ranking_table.csv` |
| Tuned nonlinear master metrics | `results/tuning/comparison/tuned_nonlinear_master_metrics.csv` |
| Tuned vs untuned comparison | `results/tuning/comparison/tuned_vs_untuned_nonlinear.csv` |
| Best hyperparameters per spec | `results/tuning/comparison/tuned_best_params_summary.csv` |
| Tuned SHAP summaries | `results/tuning/all_models/shap/plots/` |
| Residuals over time / regime errors / rolling errors | `results/tuning/all_models/error_analysis/` |
| Figure 4.1 (predicted vs actual 2024) | `results/figures/figure_4_1_predicted_vs_actual_2024.png` |
| Descriptive statistics + correlations | `results/descriptive/` |

Each experiment writes a manifest JSON capturing dataset paths, splits, feature specs, and selected parameters: `results/comparison/experiment_manifest_day3.json`, `results/tuning/experiment_manifest_tuning.json`, `results/tuning/all_models/tuned_all_followup_manifest.json`.

---

## 8. Reproducibility notes

- Random seed `42` is fixed inside Random Forest and XGBoost training.
- `pip freeze` versions are pinned in `requirements.txt`. If you need to relax them, the only versions that materially affect numerical results are `scikit-learn`, `xgboost`, `numpy`, and `shap`.
- All scripts are idempotent: rerunning a step overwrites its outputs in place rather than appending.
- The pipeline assumes UTC dates for news and exchange-local dates for market data. The merge is on calendar date, and only NASDAQ-100 trading days are retained.
- `results/sanity_checks/leakage_checks.md` is the auditable record of every alignment, fill, and lag decision in the data pipeline.

---

## 9. Known limitations

- The institutional-news universe is restricted to six outlets (Reuters, Bloomberg, CNBC, FT, Yahoo Finance, MarketWatch). Coverage gaps on weekends and holidays are encoded as zero sentiment rather than carried forward.
- The target is the index log return, not individual stocks. Inferences about AI-narrative tech stocks are motivational context, not the formal empirical target.
- SHAP values describe how the trained model uses its inputs, not causal mechanisms in the market.
- The hold-out spans a single calendar year (2024). Generalization beyond that year is not tested in this thesis.

---

## 10. Citation

If you reference this code or data pipeline, please cite the thesis:

```
[Author], "Predicting NASDAQ-100 Daily Returns with Macro Indicators and FinBERT News Sentiment,"
MSc Thesis, [Institution], 2025.
```

Third-party components retain their own licenses: FinBERT (`ProsusAI/finbert`, research use), GDELT 2.0 Event Database (CC BY 4.0 attribution), FRED data (public domain), Yahoo Finance data (terms of use apply).
