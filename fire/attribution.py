"""Campisi-style performance attribution: split a bond's (or portfolio's)
holding-period return into carry, roll-down, curve-shift and spread-change
effects, full-revaluation.

Effects are measured *marginally* from the period-start price (change one
thing at a time: only roll time forward, only shift the curve, only move the
spread) rather than sequentially chaining them. That's a deliberate choice:
sequential (waterfall) decomposition telescopes back to the total return
exactly by construction, which hides the fact that these effects don't
actually act independently on a nonlinear (convex) instrument. Measuring
them marginally leaves a small residual from the cross terms (duration x
convexity, curve-spread interaction) -- which is precisely the residual real
attribution systems report, not a bug to be zeroed out.
"""
from fire.portfolio import price_bond


def campisi_attribution(beta_start, beta_end, spread_start, spread_end,
                         maturity, coupon, dt):
    p0 = price_bond(beta_start, spread_start, maturity, coupon)

    p_roll = price_bond(beta_start, spread_start, maturity - dt, coupon)
    p_curve = price_bond(beta_end, spread_start, maturity, coupon)
    p_spread = price_bond(beta_start, spread_end, maturity, coupon)
    p_end = price_bond(beta_end, spread_end, maturity - dt, coupon)

    carry = coupon * dt
    roll_return = (p_roll - p0) / p0
    curve_return = (p_curve - p0) / p0
    spread_return = (p_spread - p0) / p0
    total_return = (p_end - p0) / p0 + carry

    residual = total_return - (carry + roll_return + curve_return + spread_return)

    return {
        "carry": carry,
        "roll_down": roll_return,
        "curve_shift": curve_return,
        "spread_change": spread_return,
        "residual": residual,
        "total_return": total_return,
        "p0": p0,
        "p_end": p_end,
    }


def portfolio_campisi_attribution(bonds, issuer_spreads_start, issuer_spreads_end,
                                   beta_start, beta_end, dt):
    """Market-value-weighted attribution across the book."""
    total_mv = 0.0
    agg = {"carry": 0.0, "roll_down": 0.0, "curve_shift": 0.0,
           "spread_change": 0.0, "residual": 0.0, "total_return": 0.0}

    for b in bonds:
        s0 = issuer_spreads_start[b["issuer_id"]]
        s1 = issuer_spreads_end[b["issuer_id"]]
        res = campisi_attribution(beta_start, beta_end, s0, s1, b["maturity"], b["coupon"], dt)
        mv = res["p0"] / 100.0 * b["notional"]
        total_mv += mv
        for k in agg:
            agg[k] += res[k] * mv

    for k in agg:
        agg[k] /= total_mv

    return agg, total_mv
