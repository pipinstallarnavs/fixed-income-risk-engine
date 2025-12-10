"""Synthetic corporate bond portfolio: issuers, bonds, and a full-revaluation
bond pricing function shared by the duration, risk, and attribution modules.
"""
import numpy as np

SECTORS = ["Financials", "Industrials", "Utilities", "Energy", "Tech", "Consumer"]
RATINGS = ["AAA", "AA", "A", "BBB", "BB"]
# rough average spread (decimal, e.g. 0.008 = 80bp) by rating, used to draw issuer spreads
RATING_SPREAD_BPS = {"AAA": 35, "AA": 55, "A": 85, "BBB": 140, "BB": 260}


def generate_issuers(n_issuers=120, seed=1):
    rng = np.random.default_rng(seed)
    issuers = []
    for i in range(n_issuers):
        sector = rng.choice(SECTORS)
        rating = rng.choice(RATINGS, p=[0.05, 0.15, 0.35, 0.30, 0.15])
        base_bp = RATING_SPREAD_BPS[rating]
        spread_level = (base_bp + rng.normal(0, base_bp * 0.15)) / 1e4  # decimal
        spread_level = max(spread_level, 0.0005)
        spread_vol = rng.uniform(0.10, 0.35) * spread_level  # daily-change scale factor
        spread_beta = rng.uniform(0.6, 1.4)  # sensitivity to the common spread factor
        issuers.append({
            "issuer_id": f"ISS{i:03d}",
            "sector": sector,
            "rating": rating,
            "spread_level": spread_level,
            "spread_vol": spread_vol,
            "spread_beta": spread_beta,
        })
    return issuers


def generate_portfolio(issuers, par_curve_tenors, par_curve_yields, seed=2):
    """One bond per issuer: random maturity, coupon set at roughly par yield
    + spread at issuance, equal-ish notional weights.
    """
    rng = np.random.default_rng(seed)
    bonds = []
    for iss in issuers:
        maturity = rng.uniform(2, 25)
        par_yield_at_mat = np.interp(maturity, par_curve_tenors, par_curve_yields)
        coupon = par_yield_at_mat + iss["spread_level"]
        notional = rng.uniform(0.5, 2.0) * 1_000_000
        bonds.append({
            "issuer_id": iss["issuer_id"],
            "maturity": maturity,
            "coupon": coupon,
            "notional": notional,
        })
    return bonds


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price_bond(nss_beta, spread, maturity, coupon, freq=2, face=100.0, curve_fn=None):
    """Full-revaluation price of a fixed-coupon bond: discount every
    remaining cashflow off the NSS zero curve plus a flat issuer spread,
    continuously compounded. `curve_fn` overrides nss_rate if you want to
    price off a bumped/shifted curve without refitting NSS (used for
    key-rate duration).
    """
    from fire.curve import nss_rate
    rate_fn = curve_fn if curve_fn is not None else (lambda tau: nss_rate(tau, nss_beta))

    n_coupons = max(int(round(maturity * freq)), 1)
    times = np.array([(k + 1) / freq for k in range(n_coupons)])
    times = times[times <= maturity + 1e-9]
    if len(times) == 0 or abs(times[-1] - maturity) > 1e-6:
        times = np.append(times, maturity)

    cfs = np.full(len(times), coupon / freq * face)
    cfs[-1] += face

    zero_rates = rate_fn(times)
    disc_rates = zero_rates + spread
    price = np.sum(cfs * np.exp(-disc_rates * times))
    return price


def portfolio_value(nss_beta, issuer_spreads, bonds, curve_fn=None):
    """Total market value of the portfolio (sum of bond prices * notional/face)."""
    total = 0.0
    for b in bonds:
        spread = issuer_spreads[b["issuer_id"]]
        p = price_bond(nss_beta, spread, b["maturity"], b["coupon"], curve_fn=curve_fn)
        total += p / 100.0 * b["notional"]
    return total
