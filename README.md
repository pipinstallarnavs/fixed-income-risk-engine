![Fixed Income Risk Engine banner](assets/banner.svg)

# Fixed Income Risk Engine

A from-scratch fixed-income factor-risk and performance-attribution pipeline built with NumPy, SciPy, and scikit-learn.

## Capabilities

- Bootstrap a Treasury zero curve from par yields
- Fit Nelson-Siegel-Svensson curves with bounded multi-start optimization
- Recover level, slope, and curvature factors with principal components
- Cluster issuers by spread behavior
- Calculate key-rate and spread duration by bump-and-reprice
- Estimate historical Value at Risk and Expected Shortfall by full revaluation
- Run Kupiec and Christoffersen backtests
- Decompose returns with Campisi-style attribution

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The runner prints a numbered report and writes a multi-panel figure to `output/report.png`.

## Repository layout

```text
fire/curve.py          Curve bootstrap, NSS fitting, simulation, and PCA
fire/portfolio.py      Issuer generation, bond pricing, and portfolio value
fire/clustering.py     Issuer peer groups
fire/duration.py       Key-rate and spread duration
fire/risk.py           Historical VaR, ES, and calibration tests
fire/attribution.py    Campisi-style return decomposition
main.py                End-to-end research pipeline
```

## Numerical design

NSS fitting separates the linear beta coefficients from the two nonlinear decay parameters and searches the latter from multiple bounded starting points. Risk and sensitivity calculations use full repricing instead of delta-gamma approximations. Attribution measures effects marginally, leaving a residual for nonlinear interactions rather than forcing an exact waterfall identity.

## Scope

Curve and spread histories are simulated, not sourced from a market-data vendor. Issuer spreads are flat across maturity, and the VaR backtest uses a fixed threshold. The project demonstrates risk mechanics and numerical controls rather than production calibration.
