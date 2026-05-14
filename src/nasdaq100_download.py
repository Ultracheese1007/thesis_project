import yfinance as yf
from pathlib import Path

# 下载 ^NDX 数据
df = yf.download("^NDX", start="2020-01-01", end="2025-01-01")

# 只保留 Close 列
df = df[["Close"]].copy()
df.columns = ["nasdaq_close"]

# 写入当前项目下的 data/raw
output_path = Path("data/raw/nasdaq100_close.csv")

# 若文件夹不存在则创建
output_path.parent.mkdir(parents=True, exist_ok=True)

# 保存
df.to_csv(output_path)

print(df.head())
print(f"Saved to: {output_path.resolve()}")