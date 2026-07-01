"""
src/eda.py
Exploratory analysis for ShiftCast.

Answers two questions before any model is built:
  1. What drives demand (shift, day, weekend, season, weather, events)?
  2. How much of the variation is predictable structure vs irreducible noise?

It prints a readable summary and saves three charts to assets/.
Run from the project root:
    python src/eda.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "sample_shifts.csv")
ASSETS = os.path.join(ROOT, "assets")
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def variance_explained(df: pd.DataFrame, by: list[str]) -> float:
    """R^2 from predicting covers with the mean of each group in `by`.
    A simple, model-free measure of how much variation those columns capture."""
    grp = df.groupby(by)["covers"].transform("mean")
    ss_res = ((df["covers"] - grp) ** 2).sum()
    ss_tot = ((df["covers"] - df["covers"].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    os.makedirs(ASSETS, exist_ok=True)

    print(f"Rows: {len(df):,}  |  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Overall covers: mean={df['covers'].mean():.1f}, "
          f"min={df['covers'].min()}, max={df['covers'].max()}\n")

    # --- group averages: which factors move demand ---
    print("Average covers by shift:")
    print(df.groupby("shift")["covers"].mean().round(1).to_string(), "\n")

    print("Average covers by day of week:")
    print(df.groupby("day_of_week")["covers"].mean()
          .reindex(DOW_ORDER).round(1).to_string(), "\n")

    for col in ["weather", "holiday", "local_event", "promotion"]:
        print(f"Average covers by {col}:")
        print(df.groupby(col)["covers"].mean().round(1).to_string(), "\n")

    # --- overdispersion: variance vs mean (Poisson assumes they are equal) ---
    print("Overdispersion check (variance / mean, Poisson would be 1.0):")
    for s in df["shift"].unique():
        x = df[df["shift"] == s]["covers"]
        print(f"  {s}: mean={x.mean():.1f}, var={x.var():.0f}, "
              f"ratio={x.var() / x.mean():.1f}")
    print()

    # --- predictable signal vs noise ---
    r2_shift_dow = variance_explained(df, ["shift", "day_of_week"])
    r2_rich = variance_explained(
        df, ["shift", "day_of_week", "weather", "holiday"])
    print("How much variation simple structure explains:")
    print(f"  shift + day_of_week:                 {r2_shift_dow * 100:.0f}%")
    print(f"  + weather + holiday:                 {r2_rich * 100:.0f}%")
    print(f"  remaining (noise + unmodeled):       {(1 - r2_rich) * 100:.0f}%\n")

    # --- chart 1: average covers by day of week, split by shift ---
    piv = (df.groupby(["day_of_week", "shift"])["covers"].mean()
           .unstack().reindex(DOW_ORDER))
    ax = piv.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Average covers by day of week and shift")
    ax.set_xlabel("")
    ax.set_ylabel("Average covers")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "covers_by_day_shift.png"), dpi=120)
    plt.close()

    # --- chart 2: distribution of covers per shift (shows the spread/noise) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in df["shift"].unique():
        ax.hist(df[df["shift"] == s]["covers"], bins=30, alpha=0.6, label=s)
    ax.set_title("Distribution of covers per shift")
    ax.set_xlabel("Covers in a shift")
    ax.set_ylabel("Number of shifts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "covers_distribution.png"), dpi=120)
    plt.close()

    # --- chart 3: covers over time (seasonality + noise) ---
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    daily = d.groupby("date")["covers"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily.index, daily.values, linewidth=0.8)
    ax.plot(daily.index, daily.rolling(14).mean(), linewidth=2,
            label="14-day average")
    ax.set_title("Total daily covers over time")
    ax.set_xlabel("")
    ax.set_ylabel("Total covers per day")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "covers_over_time.png"), dpi=120)
    plt.close()

    print(f"Saved 3 charts to {ASSETS}")


if __name__ == "__main__":
    main()
