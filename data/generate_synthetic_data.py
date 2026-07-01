"""
generate_synthetic_data.py
Synthetic shift-level demand data for ShiftCast.

No real or proprietary company data is used. Each row is ONE SHIFT (Lunch or
Dinner) on one day. For each shift we simulate the number of customers (covers)
by drawing from a Poisson process whose expected rate depends on the things that
actually move restaurant demand:

    shift, day of week, weekend, season/month, holidays, weather,
    local events, promotions, and reservations on the book.

To make the data realistic, demand is OVERDISPERSED: we multiply each shift's
rate by a random Gamma factor before the Poisson draw (a Gamma-Poisson mixture,
which is exactly a Negative Binomial). That gives variance larger than the mean,
the way real demand behaves, and it is why a Negative Binomial model is expected
to beat a plain Poisson later.

Run:
    python data/generate_synthetic_data.py                 # 1 year, 2 shifts/day
    python data/generate_synthetic_data.py --days 540 --seed 42
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SHIFTS = ["Lunch", "Dinner"]

# Baseline expected covers per shift (Dinner is the busier service).
BASE_COVERS = {"Lunch": 60.0, "Dinner": 110.0}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

# Day-of-week multiplier (Mon=0 ... Sun=6): weekends busier.
DOW_MULT = {0: 0.85, 1: 0.85, 2: 0.90, 3: 1.00,
            4: 1.30, 5: 1.45, 6: 1.10}

# Month multiplier: mild seasonality (summer patio lift, December holidays).
MONTH_MULT = {1: 0.90, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,
              7: 1.15, 8: 1.15, 9: 1.05, 10: 1.00, 11: 0.95, 12: 1.10}

# Weather is a forecast known before the shift. Bad weather lowers foot traffic.
WEATHER_TYPES = ["Clear", "Cloudy", "Rain", "Snow"]
WEATHER_MULT = {"Clear": 1.05, "Cloudy": 1.00, "Rain": 0.90, "Snow": 0.80}

# Effect sizes for binary drivers (all synthetic, easy to tune).
HOLIDAY_BOOST = 0.25
EVENT_BOOST = 0.30
PROMO_BOOST = 0.15

# Overdispersion: smaller -> more extra variance. (Gamma shape parameter.)
DISPERSION_SHAPE = 8.0

# Fraction of covers that tend to book ahead, by shift (Dinner books more).
RESERVATION_RATE = {"Lunch": 0.20, "Dinner": 0.45}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _weather_probs(month: int):
    """Seasonal weather mix: more snow in winter, more clear in summer."""
    if month in (12, 1, 2):       # winter
        return [0.30, 0.30, 0.20, 0.20]
    if month in (6, 7, 8):        # summer
        return [0.60, 0.25, 0.15, 0.00]
    return [0.45, 0.30, 0.23, 0.02]  # spring / fall


def generate(days: int = 365, seed: int = 42,
             start: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    end_date = (pd.Timestamp(start) if start
                else pd.Timestamp.today().normalize())
    start_date = end_date - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start_date, end_date, freq="D")

    rows = []
    for d in dates:
        month = d.month
        weekday = d.dayofweek
        is_weekend = int(weekday >= 5)
        # ~3% of days flagged as a (busier) holiday.
        holiday = int(rng.random() < 0.03)
        weather = str(rng.choice(WEATHER_TYPES, p=_weather_probs(month)))

        for shift in SHIFTS:
            # Events and promos vary by shift/day.
            event = int(rng.random() < (0.06 if shift == "Dinner" else 0.03))
            promo = int(rng.random() < 0.10)

            # Expected demand (the Poisson rate, lambda) from all drivers.
            lam = (BASE_COVERS[shift]
                   * DOW_MULT[weekday]
                   * MONTH_MULT[month]
                   * WEATHER_MULT[weather]
                   * (1 + HOLIDAY_BOOST * holiday)
                   * (1 + EVENT_BOOST * event)
                   * (1 + PROMO_BOOST * promo))

            # Overdispersion: Gamma multiplier with mean 1 (Gamma-Poisson = NB).
            gamma = rng.gamma(shape=DISPERSION_SHAPE, scale=1.0 / DISPERSION_SHAPE)
            covers = int(rng.poisson(lam * gamma))

            # Reservations on the book: a noisy fraction of true demand,
            # known before the shift. A genuine predictor, not the target.
            res_lam = covers * RESERVATION_RATE[shift]
            reservations = int(rng.poisson(max(res_lam, 0.1)))

            rows.append({
                "date": d.date().isoformat(),
                "day_of_week": DAYS[weekday],
                "is_weekend": is_weekend,
                "month": month,
                "season": _season(month),
                "shift": shift,
                "holiday": holiday,
                "weather": weather,
                "local_event": event,
                "promotion": promo,
                "reservations": reservations,
                "covers": covers,          # <-- TARGET (simulated)
            })

    df = pd.DataFrame(rows)

    # Predictor: same shift one week ago (known in advance).
    df = df.sort_values(["shift", "date"]).reset_index(drop=True)
    df["covers_last_week"] = df.groupby("shift")["covers"].shift(7)

    # Order columns and restore chronological order.
    cols = ["date", "day_of_week", "is_weekend", "month", "season", "shift",
            "holiday", "weather", "local_event", "promotion", "reservations",
            "covers_last_week", "covers"]
    return df[cols].sort_values(["date", "shift"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic ShiftCast data.")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=str, default=None,
                   help="End date (YYYY-MM-DD); data covers the days before it.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    df = generate(days=args.days, seed=args.seed, start=args.start)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_shifts.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df):,} shift rows to {out}")
    print(f"Date range : {df['date'].min()} to {df['date'].max()}")
    print(f"Shifts/day : {df['shift'].nunique()}  ({', '.join(SHIFTS)})")


if __name__ == "__main__":
    main()
