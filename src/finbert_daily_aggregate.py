from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RAW_PATH = Path("data/raw/financial_news_2020_2024_full.csv")
OUTPUT_PATH = Path("data/processed/daily/daily_sentiment.csv")
LOG_PATH = Path("logs/daily_finbert_aggregate_summary.json")

MODEL_NAME = "ProsusAI/finbert"

# ===== Runtime configuration =====
CSV_CHUNK_SIZE = 20000
INFERENCE_BATCH_SIZE = 64
MAX_LENGTH = 256

# Smoke test:
# 设为 None 表示跑全量
MAX_CHUNKS = None

# Thesis analysis window
START_DATE = pd.Timestamp("2020-01-01", tz="UTC")
END_DATE = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

# 自动识别常见列名
DATE_CANDIDATES = ["date", "published_at", "publish_date", "datetime", "time", "timestamp"]
TITLE_CANDIDATES = ["title", "headline"]
TEXT_CANDIDATES = ["text", "content", "article", "body", "summary"]


def detect_columns(columns: List[str]) -> Tuple[str, str | None, str | None]:
    colset = set(columns)

    date_col = next((c for c in DATE_CANDIDATES if c in colset), None)
    title_col = next((c for c in TITLE_CANDIDATES if c in colset), None)
    text_col = next((c for c in TEXT_CANDIDATES if c in colset), None)

    if date_col is None:
        raise KeyError(
            f"Could not detect a date column. Available columns: {columns}"
        )

    if title_col is None and text_col is None:
        raise KeyError(
            f"Could not detect a text/title column. Available columns: {columns}"
        )

    return date_col, title_col, text_col


def build_input_text(df: pd.DataFrame, title_col: str | None, text_col: str | None) -> pd.Series:
    title = df[title_col].fillna("").astype(str) if title_col else pd.Series([""] * len(df), index=df.index)
    text = df[text_col].fillna("").astype(str) if text_col else pd.Series([""] * len(df), index=df.index)

    combined = (title.str.strip() + ". " + text.str.strip()).str.strip(" .")
    combined = combined.replace("", np.nan)
    return combined


def load_finbert():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    id2label = {int(k): str(v).lower() for k, v in id2label.items()}

    label_to_idx = {label: idx for idx, label in id2label.items()}
    required = {"positive", "negative", "neutral"}
    if not required.issubset(label_to_idx.keys()):
        raise ValueError(f"Unexpected FinBERT labels: {label_to_idx}")

    return tokenizer, model, device, label_to_idx


@torch.no_grad()
def infer_scores(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    label_to_idx: Dict[str, int],
) -> np.ndarray:
    scores_all = []

    for start in range(0, len(texts), INFERENCE_BATCH_SIZE):
        batch_texts = texts[start:start + INFERENCE_BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        pos = probs[:, label_to_idx["positive"]]
        neg = probs[:, label_to_idx["negative"]]

        scores = pos - neg
        scores_all.append(scores)

    return np.concatenate(scores_all, axis=0)


def update_daily_stats(
    chunk_days: pd.Series,
    scores: np.ndarray,
    stats: Dict[pd.Timestamp, Dict[str, float]],
) -> None:
    temp = pd.DataFrame({
        "day": chunk_days.values,
        "score": scores,
    })

    temp["score_sq"] = temp["score"] ** 2
    temp["is_extreme"] = (temp["score"].abs() > 0.8).astype(int)

    grouped = temp.groupby("day").agg(
        article_count=("score", "size"),
        score_sum=("score", "sum"),
        score_sq_sum=("score_sq", "sum"),
        extreme_count=("is_extreme", "sum"),
    )

    for day, row in grouped.iterrows():
        if day not in stats:
            stats[day] = {
                "article_count": 0,
                "score_sum": 0.0,
                "score_sq_sum": 0.0,
                "extreme_count": 0,
            }

        stats[day]["article_count"] += int(row["article_count"])
        stats[day]["score_sum"] += float(row["score_sum"])
        stats[day]["score_sq_sum"] += float(row["score_sq_sum"])
        stats[day]["extreme_count"] += int(row["extreme_count"])


def finalize_daily_stats(stats: Dict[pd.Timestamp, Dict[str, float]]) -> pd.DataFrame:
    rows = []

    for day, s in stats.items():
        n = s["article_count"]
        mean = s["score_sum"] / n
        variance = max(s["score_sq_sum"] / n - mean ** 2, 0.0)
        std = variance ** 0.5
        extreme_ratio = s["extreme_count"] / n

        rows.append({
            "date": pd.to_datetime(day),
            "sentiment_mean": mean,
            "sentiment_std": std,
            "sentiment_extreme": extreme_ratio,
            "article_count": n,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(RAW_PATH, nrows=5)
    date_col, title_col, text_col = detect_columns(header.columns.tolist())

    print(f"[INFO] Detected date column: {date_col}")
    print(f"[INFO] Detected title column: {title_col}")
    print(f"[INFO] Detected text column: {text_col}")
    print(f"[INFO] Filtering analysis window: {START_DATE} -> {END_DATE}")

    tokenizer, model, device, label_to_idx = load_finbert()
    print(f"[INFO] Using device: {device}")

    stats: Dict[pd.Timestamp, Dict[str, float]] = {}
    total_rows_seen = 0
    total_rows_used = 0
    chunk_count = 0

    usecols = [date_col]
    if title_col:
        usecols.append(title_col)
    if text_col and text_col not in usecols:
        usecols.append(text_col)

    for chunk in pd.read_csv(RAW_PATH, chunksize=CSV_CHUNK_SIZE, usecols=usecols):
        chunk_count += 1

        if MAX_CHUNKS is not None and chunk_count > MAX_CHUNKS:
            print(f"[INFO] Reached MAX_CHUNKS={MAX_CHUNKS}. Stopping early.")
            break

        total_rows_seen += len(chunk)

        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk = chunk.dropna(subset=[date_col]).copy()

        # 只保留 thesis 需要的时间范围
        chunk = chunk[
            (chunk[date_col] >= START_DATE) &
            (chunk[date_col] <= END_DATE)
        ].copy()

        if chunk.empty:
            print(f"[INFO] Chunk {chunk_count}: skipped (out of date range or empty)")
            continue

        chunk["model_text"] = build_input_text(chunk, title_col, text_col)
        chunk = chunk.dropna(subset=["model_text"]).copy()

        if chunk.empty:
            print(f"[INFO] Chunk {chunk_count}: skipped (empty after cleaning)")
            continue

        chunk["day"] = chunk[date_col].dt.floor("D")
        texts = chunk["model_text"].tolist()

        scores = infer_scores(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_to_idx=label_to_idx,
        )

        update_daily_stats(chunk["day"], scores, stats)

        total_rows_used += len(chunk)
        print(
            f"[INFO] Chunk {chunk_count}: processed {len(chunk)} usable rows "
            f"(cumulative usable rows: {total_rows_used})"
        )

    if not stats:
        raise RuntimeError("No daily statistics were generated. Check input columns and text availability.")

    df_daily = finalize_daily_stats(stats)
    df_daily.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "raw_file": str(RAW_PATH),
        "output_file": str(OUTPUT_PATH),
        "model_name": MODEL_NAME,
        "device": device,
        "csv_chunk_size": CSV_CHUNK_SIZE,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_chunks": MAX_CHUNKS,
        "start_date": str(START_DATE),
        "end_date": str(END_DATE),
        "date_column": date_col,
        "title_column": title_col,
        "text_column": text_col,
        "total_rows_seen": total_rows_seen,
        "total_rows_used": total_rows_used,
        "n_daily_observations": int(len(df_daily)),
        "date_min": str(df_daily["date"].min()) if not df_daily.empty else None,
        "date_max": str(df_daily["date"].max()) if not df_daily.empty else None,
    }

    LOG_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved daily sentiment to: {OUTPUT_PATH}")
    print(f"[OK] Saved run summary to: {LOG_PATH}")
    print("[OK] Preview:")
    print(df_daily.head().to_string(index=False))


if __name__ == "__main__":
    main()
