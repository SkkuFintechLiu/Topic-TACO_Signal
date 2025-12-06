import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# === 1. Load tweets feature table ===
df = pd.read_csv("/Users/liuhongchuan/Documents/trae_projects/Topic-TACO_Signal/result/tweets_features.csv")

# Convert window_center to datetime and sort
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
df = df.sort_values("date")

# Extract time series
df["abs_signal"] = pd.to_numeric(df["taco_signal"], errors="coerce").abs()
idx = df.groupby("date")["abs_signal"].idxmax()
daily = df.loc[idx].sort_values("date")
ts_date = daily["date"]
_raw = pd.to_numeric(daily["taco_signal"], errors="coerce")
_mean = float(np.nanmean(_raw.values)) if len(_raw) else 0.0
_std = float(np.nanstd(_raw.values)) if len(_raw) else 1.0
if np.isfinite(_std) and _std > 0:
    daily["taco_z"] = (_raw - _mean) / _std
else:
    daily["taco_z"] = 0.0
ts_signal = daily["taco_z"]

# Clip time range to requested window
START = pd.to_datetime("2017-01-20")
END = pd.to_datetime("2021-01-21")
mask = (ts_date >= START) & (ts_date <= END)
ts_date = ts_date[mask]
ts_signal = ts_signal[mask]

q95 = ts_signal.quantile(0.95)
q05 = ts_signal.quantile(0.05)

# === 2. Load deduped extreme events ===
ev = pd.read_csv("/Users/liuhongchuan/Documents/trae_projects/Topic-TACO_Signal/result/events_deduped.csv")
ev["window_center"] = pd.to_datetime(ev["window_center"], errors="coerce").dt.tz_localize(None).dt.normalize()
ev = ev[(ev["window_center"] >= START) & (ev["window_center"] <= END)]

pos = ev[ev["direction"] == "pos"]
neg = ev[ev["direction"] == "neg"]

# === 3. Plot ===
fig, ax = plt.subplots(figsize=(16, 6))

# main line
ax.plot(ts_date, ts_signal, color="orange", linewidth=1.2, label="TACO-Signal")

# thresholds
ax.axhline(q95, color="pink", linestyle="--", linewidth=1.0)
ax.axhline(q05, color="cornflowerblue", linestyle="--", linewidth=1.0)

# POS events
ax.scatter(pos["window_center"], pos["peak_score"], color="red", s=40, label="Top POS")
for _, r in pos.iterrows():
    ax.text(r["window_center"], r["peak_score"] + 0.03,
            r["window_center"].strftime("%Y-%m-%d"), color="red", fontsize=8, rotation=30)

# NEG events
ax.scatter(neg["window_center"], neg["peak_score"], color="blue", s=40, label="Top NEG")
for _, r in neg.iterrows():
    ax.text(r["window_center"], r["peak_score"] - 0.05,
            r["window_center"].strftime("%Y-%m-%d"), color="blue", fontsize=8, rotation=30)

ax.set_title("TACO-Signal Time Series with POS & NEG Extreme Events\n(2017-01-20 ~ 2021-01-21)")
ax.set_xlabel("Date")
ax.set_ylabel("Standardized TACO-Signal (z-score)")
ax.grid(True, linestyle=":", linewidth=0.7)
ax.legend()

plt.tight_layout()
out_dir = "/Users/liuhongchuan/Documents/trae_projects/Topic-TACO_Signal/super_taco"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "taco_signal_timeseries.png")
out_svg = os.path.join(out_dir, "taco_signal_timeseries.svg")
plt.savefig(out_png, dpi=150)
plt.savefig(out_svg)
plt.close()
