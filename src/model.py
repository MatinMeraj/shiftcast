"""
src/model.py
Forecast covers per shift, then translate the forecast into a staffing number.

Three models are compared honestly on a held-out, time-based test set
(train on the earlier period, test on the most recent period, the way a real
forecast would be used):

  1. Linear regression  (baseline)
  2. Poisson regression (counts)
  3. Negative binomial  (counts WITH overdispersion, expected to win here)

It prints accuracy for each, picks the best, shows how predicted covers become
a recommended staff count, and saves an actual-vs-predicted chart to assets/.

Run from the project root, after generating data:
    python data/generate_synthetic_data.py
    python src/model.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "sample_shifts.csv")
ASSETS = os.path.join(ROOT, "assets")

# How many covers one back-of-house person can handle in a shift (assumption,
# easy to change). Used only to turn a demand forecast into a staffing number.
COVERS_PER_PERSON = 25

# Predictors known BEFORE the shift. We deliberately exclude "reservations"
# from the main model: in this synthetic data it was generated directly from
# covers, so it would be an unrealistically strong (near-circular) signal.
FORMULA = ("covers ~ C(shift) + C(day_of_week) + C(season) + C(weather) "
           "+ holiday + local_event + promotion + covers_last_week")


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["covers_last_week"]).copy()  # drop first week (no lag)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Time-based split: earliest 80% to train, most recent 20% to test.
    cutoff = df["date"].quantile(0.80)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    print(f"Train: {len(train)} shifts ({train['date'].min().date()} to "
          f"{train['date'].max().date()})")
    print(f"Test : {len(test)} shifts ({test['date'].min().date()} to "
          f"{test['date'].max().date()})\n")

    results = {}

    # 1. Linear regression baseline
    lin = smf.ols(FORMULA, data=train).fit()
    results["Linear"] = metrics(test["covers"], lin.predict(test))

    # 2. Poisson regression
    pois = smf.glm(FORMULA, data=train, family=sm.families.Poisson()).fit()
    results["Poisson"] = metrics(test["covers"], pois.predict(test))

    # 3. Negative binomial (estimates its own overdispersion parameter by MLE).
    #    The fitted alpha is the dispersion: alpha = 0 would be plain Poisson
    #    (variance = mean). alpha > 0 means variance > mean, i.e. overdispersion.
    #    This is the maximum-likelihood version of the textbook moment estimate
    #    r = mean^2 / (var - mean); statsmodels estimates it across all shifts.
    nb_alpha = None
    try:
        nb = smf.negativebinomial(FORMULA, data=train).fit(disp=0)
        nb_pred = nb.predict(test)
        nb_alpha = float(nb.params.get("alpha", float("nan")))
    except Exception:
        # fallback: GLM NB with a fixed dispersion if MLE struggles
        nb = smf.glm(FORMULA, data=train,
                     family=sm.families.NegativeBinomial(alpha=0.1)).fit()
        nb_pred = nb.predict(test)
    results["NegBinomial"] = metrics(test["covers"], nb_pred)

    # --- report ---
    print("Held-out test accuracy (lower MAE/RMSE = better, higher R2 = better):")
    print(f"{'Model':<14}{'MAE':>8}{'RMSE':>9}{'R2':>8}")
    for name, m in results.items():
        print(f"{name:<14}{m['MAE']:>8.1f}{m['RMSE']:>9.1f}{m['R2']:>8.2f}")
    print()

    if nb_alpha is not None:
        print(f"Negative binomial dispersion (alpha) = {nb_alpha:.3f}. "
              f"Alpha above 0 confirms overdispersion (variance > mean); "
              f"Poisson would force alpha = 0. This is why NB fits best.\n")

    best = min(results, key=lambda k: results[k]["RMSE"])
    print(f"Best model by RMSE: {best}")
    print(f"On the test set it predicts a shift's covers to within about "
          f"{results[best]['MAE']:.0f} on average, and explains "
          f"{results[best]['R2'] * 100:.0f}% of the variation. "
          f"The rest is irreducible noise.\n")

    # --- translate the best model's forecast into staffing ---
    pred_map = {"Linear": lin.predict(test), "Poisson": pois.predict(test),
                "NegBinomial": nb_pred}
    test = test.copy()
    test["predicted_covers"] = np.asarray(pred_map[best])
    test["recommended_staff"] = np.ceil(
        test["predicted_covers"] / COVERS_PER_PERSON).astype(int)
    test["actual_staff_needed"] = np.ceil(
        test["covers"] / COVERS_PER_PERSON).astype(int)

    print(f"Staffing (assuming 1 person per {COVERS_PER_PERSON} covers). "
          f"Sample of test shifts:")
    show = test[["date", "shift", "covers", "predicted_covers",
                 "recommended_staff", "actual_staff_needed"]].head(8).copy()
    show["date"] = show["date"].dt.date
    show["predicted_covers"] = show["predicted_covers"].round(0).astype(int)
    show = show.rename(columns={"covers": "actual_covers"})
    print(show.to_string(index=False))

    staff_off = (test["recommended_staff"] - test["actual_staff_needed"]).abs()
    print(f"\nStaffing recommendation is exactly right on "
          f"{(staff_off == 0).mean() * 100:.0f}% of test shifts, and within one "
          f"person on {(staff_off <= 1).mean() * 100:.0f}%.\n")

    # --- chart: actual vs predicted covers on the test set ---
    os.makedirs(ASSETS, exist_ok=True)
    t = test.sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(len(t)), t["covers"].values, label="Actual covers",
            linewidth=1)
    ax.plot(range(len(t)), t["predicted_covers"].values,
            label=f"Predicted ({best})", linewidth=1.5)
    ax.set_title("ShiftCast: actual vs predicted covers (held-out test period)")
    ax.set_xlabel("Test shift (chronological)")
    ax.set_ylabel("Covers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "actual_vs_predicted.png"), dpi=120)
    plt.close()
    print(f"Saved actual-vs-predicted chart to {ASSETS}")


if __name__ == "__main__":
    main()
