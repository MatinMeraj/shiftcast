# ShiftCast

ShiftCast predicts how busy each restaurant shift will be, so a high-volume
restaurant can staff to real demand instead of guessing. The goal is to reduce
the overtime and burnout that drive back-of-house turnover.

> **Status: working prototype.** Built on synthetic data, not real or
> proprietary company data.

## Background

I work as a cook (back of house) at a high-volume restaurant. The kitchen has
been losing back-of-house staff, especially newer hires. Talking to managers,
the recurring pain is scheduling: it is hard to know how many people each shift
needs, so the team often keeps people overtime to cover busy periods.

## The assumption (stated openly)

The staffing pressure comes mainly from keeping employees overtime because
shifts are understaffed relative to actual demand. This is a hypothesis to test,
not a proven fact. Treating it as an explicit assumption (rather than a
conclusion) is a deliberate experimental-design choice, so the project stays
honest about what it does and does not show.

## Business problem vs. analytical problem

**Business problem:** the restaurant cannot reliably predict how busy each shift
will be, so it over- or understaffs. That drives overtime cost and burnout,
which contribute to turnover.

**Analytical problem:** predict the number of customers (covers) per shift from
variables known before the shift, and measure how much of demand is predictable
versus irreducible noise.

**The key distinction:** model accuracy is the technical objective, but the real
objective is the business outcome (reducing overtime-driven churn by staffing to
demand). The prediction is the mechanism, not the goal.

## Data

A synthetic data generator simulates customer arrivals as a Poisson process
whose rate varies by shift, day of week, weekend, and season, with extra
variance added so the data is overdispersed like real demand. The data is
clearly labeled synthetic. No real or proprietary company data is used.

## Target and predictors

**Target (Y):** customers (covers) per shift.

**Predictors (X), all known before the shift:** day of week, shift, weekend
flag, month or season, holiday flag, weather, local events, promotions, and the
prior week's same shift. (Reservations are generated and stored but kept out of
the main model, since in this synthetic data they are derived from covers and
would be an unrealistically strong signal.)

## Method

- **Baseline:** linear regression.
- **Core model:** Poisson regression, since covers per shift is count data.
- **Refinement:** negative binomial regression, for overdispersion (variance
  larger than the mean, which this demand has).
- **Evaluation:** a time-based split (train on the earlier period, test on the
  most recent period), scored by MAE, RMSE, and R2 on the held-out test set.
- **Output:** predicted covers per shift, converted into a recommended staff
  count, plus an honest statement of how much of demand is predictable.

## Results

On the held-out test period (train on the earlier roughly ten months, predict
the most recent roughly two months):

- The negative binomial model predicts covers per shift to within about 28 on
  average and explains roughly 53% of the variation. The remaining roughly 47%
  is irreducible noise (walk-ins, random busy nights), matching the exploratory
  finding.
- Negative binomial slightly beats Poisson and linear regression, as expected
  given the demand is overdispersed.
- Translated to staffing (one person per 25 covers), the recommendation is
  exactly right on about 27% of shifts and within one person on about 71%.

As an advanced cross-check, a state-space (Kalman) time-series forecaster was
also tried (`forecast_kalman.py`). It beats a naive "same shift last week"
baseline but does not beat the feature-based regression here, because most of
the predictable signal comes from known calendar and weather conditions rather
than from momentum in the series. That is itself a useful finding: a regression
on known drivers is the right tool for this problem, while a Kalman filter is
better suited when a series carries its own momentum.

**The honest conclusion:** about half of shift demand is predictable and should
be staffed to; the other half is not. So the right operational response is to
staff the predictable base and keep flexible capacity (on-call, cross-training)
for the rest, rather than chase a perfect forecast that cannot exist.

## Project structure

```
shiftcast/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── generate_synthetic_data.py
│   └── sample_shifts.csv
├── src/
│   ├── eda.py
│   ├── model.py
│   └── forecast_kalman.py
└── assets/
    (charts produced by eda.py and model.py)
```

## How to run

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py   # writes data/sample_shifts.csv
python src/eda.py                         # summary + charts in assets/
python src/model.py                       # model comparison + staffing
python src/forecast_kalman.py              # time-series (Kalman) comparison
```

## Roadmap

1. Define the problem and method (this README). Done.
2. Synthetic data generator (Poisson arrivals by shift, day, season). Done.
3. Exploratory analysis: what drives demand, and how noisy is it. Done.
4. Models: linear baseline, then Poisson and negative binomial. Done.
5. Translate predictions into a staffing recommendation. Done.

Extensions explored: a Kalman state-space forecaster (`forecast_kalman.py`).
Possible further work: hourly forecasts, live weather-forecast integration, and
a cost model that weighs overstaffing against understaffing.

## Disclaimer

This is an independent learning project built on synthetic data. It does not use
real company data, and its outputs are illustrative, not operational advice.
