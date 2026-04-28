# Fixed Income Factor Risk & Performance Attribution Engine

A from-scratch (numpy/scipy/sklearn, no QuantLib) pipeline that:

1. **Bootstraps** a Treasury zero curve from par yields at key tenors.
2. **Fits Nelson-Siegel-Svensson** to the zero curve via a multi-start grid
   over the two decay parameters (lambda1, lambda2), since those enter the
   model nonlinearly and are notoriously hard to identify from a single
   starting guess — an 8x8 grid (64 starts) is used, and for each fixed
   (lambda1, lambda2) the remaining four parameters are solved in closed
   form (ordinary least squares), so the search is only ever over a 2D
   surface.
3. Simulates a history of daily curve moves and runs **PCA** on them to
   recover the standard level / slope / curvature factor structure.
4. Generates a synthetic universe of 120 issuers and one bond each, and
   **k-means clusters** issuers into spread-behaviour peer groups.
5. Computes **key-rate duration** (bucketed 1/2/5/10/20/30y) and **spread
   duration** by bump-and-reprice full revaluation.
6. Computes **1-day 99% historical VaR / Expected Shortfall** by full
   revaluation over 1,000 simulated historical (curve, spread) scenarios,
   then backtests the VaR estimate with the **Kupiec proportion-of-failures**
   and **Christoffersen independence** likelihood-ratio tests.
7. Runs a **Campisi-style attribution**, splitting the portfolio's
   one-month return into carry, roll-down, curve-shift and spread-change
   effects, full revaluation.

There's no real market data feed here — curve and spread histories are
simulated (level/slope/curvature factor shocks + issuer-specific spread
processes) so the whole thing runs standalone. The point was to actually
implement the mechanics correctly, not to source real data.

## Run it

```bash
cd ~/Programming/fixed-income-risk-engine
source venv/bin/activate
python main.py
```

Prints a numbered report to stdout and saves `output/report.png` (curve fit,
PCA scree, cluster scatter, KRD bar chart, loss histogram with VaR/ES,
attribution waterfall).

## Layout

```
fire/
  curve.py        bootstrap, NSS fit, historical curve sim, PCA
  portfolio.py     issuers, bonds, full-revaluation bond pricing
  clustering.py    k-means peer groups
  duration.py      key-rate duration (tent-bump reprice), spread duration
  risk.py          historical VaR/ES, Kupiec test, Christoffersen test
  attribution.py   Campisi waterfall
main.py            orchestrates everything, prints report, saves charts
```

## Things worth understanding before an interview asks about them

- **Why 64 starts, not 1.** With (lambda1, lambda2) fixed, NSS is *linear*
  in beta0..beta3, so that inner fit is closed-form OLS — see
  `_fit_nss_given_lambdas`. The nonconvexity lives entirely in the 2D
  (lambda1, lambda2) space, which is what the grid search covers. Try
  deleting the multi-start and fitting from one bad guess (e.g. lambda1=lambda2)
  to see it land in a degenerate solution — that's the "tau
  non-identifiability" the resume line refers to. Also note the optimizer is
  *bounded* (`L-BFGS-B`, lambda in [0.1, 30]); an earlier unbounded version
  let lambda1 wander to ~32,000, made two of the NSS basis functions nearly
  collinear, and produced a numerically "perfect" but meaningless fit with
  beta coefficients in the hundreds of thousands that canceled each other out.
- **Campisi residual is real, not decorative.** The four effects are
  computed *marginally* (change one input at a time from the period-start
  price), not sequentially. Sequential/waterfall decomposition telescopes
  back to the total return exactly by construction — which would make the
  "residual under 3bps" line on the resume meaningless, since it'd always
  be exactly zero. Marginal decomposition leaves a genuine residual from
  cross terms (duration x convexity, curve-spread interaction), which is
  what real attribution systems report and explain away as convexity.
- **The VaR backtest can genuinely fail, and that's the point.** Run
  `main.py` a few times with different seeds in the VaR-estimation vs.
  backtest windows (see the `rng_seed` offsets in `main.py`) and you'll see
  the Kupiec test sometimes reject calibration (e.g. ~15-22 exceptions
  instead of ~10 over 1,000 days). That's not a bug: a 99% VaR estimated
  from only 1,000 historical scenarios is an order-statistic estimate of a
  1-in-100 tail, informed by roughly the bottom 10 observations — its
  sampling error is large. That's exactly why banks run POF/independence
  backtests in production rather than trusting a single VaR estimate.
- **Everything is full revaluation, not delta/gamma approximation.** Key-rate
  duration, spread duration, VaR/ES and attribution all reprice bonds
  exactly off a bumped or shifted curve (`portfolio.price_bond`) rather than
  approximating with duration/convexity sensitivities. That's slower but
  it's the honest version of what the resume bullets claim.

## Known simplifications (say so if asked)

- VaR backtest uses a **fixed** VaR threshold rather than a rolling
  re-estimation window, to keep runtime reasonable (a full rolling backtest
  would be ~1000x more repricing). Noted inline in `main.py`.
- Spreads are flat per issuer (no term structure of credit spread).
- No real market data — curve/spread histories are simulated factor
  processes, not pulled from a vendor feed.
