# ============================================================
# GDELT Daily Institutional News Collector
# Period: 2024-03-05 → 2024-12-31
# Output schema: date | text
# ============================================================

import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta
from tqdm import tqdm
import trafilatura
import os
import time

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BASE_URL = "http://data.gdeltproject.org/gdeltv2/"
START_DATE = "2024-03-05"
END_DATE   = "2024-12-31"

OUT_DIR = "gdelt_daily_news"
os.makedirs(OUT_DIR, exist_ok=True)

INST_PATTERN = (
    "reuters.com|bloomberg.com|cnbc.com|"
    "finance.yahoo.com|ft.com|marketwatch.com"
)

MIN_TEXT_LEN = 200
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.2  # polite crawling

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def time_slots_for_day(date_obj):
    return [
        (date_obj + timedelta(minutes=15*i)).strftime("%Y%m%d%H%M%S")
        for i in range(96)
    ]

def collect_one_day(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    out_file = os.path.join(
        OUT_DIR,
        f"gdelt_news_{date_str}.csv"
    )

    if os.path.exists(out_file):
        print(f"[SKIP] {date_str} already collected.")
        return

    rows = []

    print(f"[START] Collecting {date_str}")

    for ts in tqdm(time_slots_for_day(date_obj), leave=False):
        url = f"{BASE_URL}{ts}.export.CSV.zip"

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep="\t", header=None, low_memory=False)

            # SQLDATE (index=1), SOURCEURL (index=60)
            df = df[[1, 60]]
            df.columns = ["date", "url"]

            df = df[
                df["url"]
                .astype(str)
                .str.contains(INST_PATTERN, case=False, na=False)
            ]

            for _, row in df.iterrows():
                article_date = pd.to_datetime(
                    str(row["date"]),
                    format="%Y%m%d",
                    utc=True
                )

                downloaded = trafilatura.fetch_url(row["url"])
                if downloaded is None:
                    continue

                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False
                )

                if text is None or len(text) < MIN_TEXT_LEN:
                    continue

                rows.append({
                    "date": article_date,
                    "text": text
                })

                time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception:
            continue

    if len(rows) == 0:
        print(f"[WARN] No articles collected for {date_str}")
        return

    df_out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["date", "text"])
        .reset_index(drop=True)
    )

    df_out.to_csv(out_file, index=False)
    print(f"[DONE] {date_str} → {len(df_out)} articles saved.")

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
current = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")

while current <= end:
    collect_one_day(current.strftime("%Y-%m-%d"))
    current += timedelta(days=1)
