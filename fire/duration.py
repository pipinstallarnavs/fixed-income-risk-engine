"""Key-rate duration and spread duration via bump-and-reprice (full
revaluation), plus portfolio-level aggregation.
"""
import numpy as np
from fire.curve import nss_rate
from fire.portfolio import price_bond

KEY_RATE_TENORS = [1, 2, 5, 10, 20, 30]


def _tent_weight(tau, key_tenors, k):
    """Triangular 'bucket' shape used to bump a single key-rate node: 1 at
    node k, decaying linearly to 0 at its neighbours, flat outside the
    curve's endpoints (so the 1y bucket also covers everything shorter than
    1y, and the 30y bucket covers anything longer).
    """
    tau = np.asarray(tau, dtype=float)
    node = key_tenors[k]
    left = key_tenors[k - 1] if k > 0 else -np.inf
    right = key_tenors[k + 1] if k < len(key_tenors) - 1 else np.inf

    w = np.zeros_like(tau)
    rising = (tau >= left) & (tau <= node)
    w[rising] = 1.0 if left == -np.inf else (tau[rising] - left) / (node - left)
    falling = (tau > node) & (tau <= right)
    w[falling] = 1.0 if right == np.inf else (right - tau[falling]) / (right - node)
    return w


def key_rate_durations(nss_beta, spread, maturity, coupon, key_tenors=None, bump=1e-4):
    key_tenors = key_tenors or KEY_RATE_TENORS
    base_price = price_bond(nss_beta, spread, maturity, coupon)

    krds = {}
    for k, node in enumerate(key_tenors):
        def curve_up(tau, k=k):
            return nss_rate(tau, nss_beta) + bump * _tent_weight(tau, key_tenors, k)

        def curve_down(tau, k=k):
            return nss_rate(tau, nss_beta) - bump * _tent_weight(tau, key_tenors, k)

        p_up = price_bond(nss_beta, spread, maturity, coupon, curve_fn=curve_up)
        p_down = price_bond(nss_beta, spread, maturity, coupon, curve_fn=curve_down)
        krds[node] = -(p_up - p_down) / (2 * bump * base_price)

    return krds, base_price


def spread_duration(nss_beta, spread, maturity, coupon, bump=1e-4):
    p0 = price_bond(nss_beta, spread, maturity, coupon)
    p_up = price_bond(nss_beta, spread + bump, maturity, coupon)
    p_down = price_bond(nss_beta, spread - bump, maturity, coupon)
    return -(p_up - p_down) / (2 * bump * p0)


def portfolio_durations(nss_beta, issuer_spreads, bonds, key_tenors=None):
    """Market-value-weighted key-rate and spread durations across the book."""
    key_tenors = key_tenors or KEY_RATE_TENORS
    total_mv = 0.0
    agg_krd = {t: 0.0 for t in key_tenors}
    agg_spread_dur = 0.0

    for b in bonds:
        spread = issuer_spreads[b["issuer_id"]]
        krds, price = key_rate_durations(nss_beta, spread, b["maturity"], b["coupon"], key_tenors)
        sdur = spread_duration(nss_beta, spread, b["maturity"], b["coupon"])
        mv = price / 100.0 * b["notional"]

        total_mv += mv
        for t in key_tenors:
            agg_krd[t] += krds[t] * mv
        agg_spread_dur += sdur * mv

    for t in key_tenors:
        agg_krd[t] /= total_mv
    agg_spread_dur /= total_mv

    return agg_krd, agg_spread_dur, total_mv
