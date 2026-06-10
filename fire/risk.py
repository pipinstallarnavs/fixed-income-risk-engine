"""Historical-simulation VaR / Expected Shortfall via full revaluation, plus
Kupiec proportion-of-failures and Christoffersen independence backtests.
"""
import numpy as np
from scipy import stats

from fire.curve import simulate_historical_curves, nss_rate
from fire.portfolio import portfolio_value

SCENARIO_TENORS = [1, 2, 5, 10, 20, 30]


def simulate_spread_changes(issuers, n_days, seed=0, common_vol=0.35e-4):
    """Daily issuer spread changes = common factor (systematic spread
    widening/tightening, e.g. credit-cycle risk-on/risk-off) * issuer beta,
    plus idiosyncratic noise scaled by each issuer's own spread vol.
    """
    rng = np.random.default_rng(seed)
    n_issuers = len(issuers)
    common = rng.normal(0, common_vol, n_days)
    betas = np.array([iss["spread_beta"] for iss in issuers])
    vols = np.array([iss["spread_vol"] for iss in issuers])
    idio = rng.normal(0, 1, (n_days, n_issuers)) * vols
    changes = np.outer(common, betas) + idio
    return changes  # (n_days, n_issuers), decimal


def _scenario_curve_fn(nss_beta, tenor_nodes, deltas):
    def f(tau):
        interp_delta = np.interp(tau, tenor_nodes, deltas)
        return nss_rate(tau, nss_beta) + interp_delta
    return f


def scenario_pnl(nss_beta, issuer_spreads, bonds, base_value,
                  curve_changes, spread_changes, issuer_ids):
    """Full revaluation of the portfolio under each of n_scenarios historical
    (curve delta, spread delta) draws. Returns an array of P&L, one per
    scenario.
    """
    n_scenarios = curve_changes.shape[0]
    pnl = np.zeros(n_scenarios)

    for s in range(n_scenarios):
        curve_fn = _scenario_curve_fn(nss_beta, SCENARIO_TENORS, curve_changes[s])
        bumped_spreads = {
            iid: issuer_spreads[iid] + spread_changes[s, j]
            for j, iid in enumerate(issuer_ids)
        }
        new_value = portfolio_value(nss_beta, bumped_spreads, bonds, curve_fn=curve_fn)
        pnl[s] = new_value - base_value

    return pnl


def historical_var_es(pnl, confidence=0.99):
    """1-day historical VaR / ES from a P&L (not return) array. VaR and ES
    are reported as positive loss numbers.
    """
    losses = -pnl
    var = np.percentile(losses, confidence * 100)
    tail = losses[losses >= var]
    es = tail.mean() if len(tail) > 0 else var
    return var, es


# ---------------------------------------------------------------------------
# Backtesting: Kupiec POF + Christoffersen independence
# ---------------------------------------------------------------------------

def kupiec_pof_test(exceptions, p=0.01):
    """Likelihood-ratio test for whether the observed exception rate matches
    the VaR confidence level. exceptions: bool array, True on days the loss
    breached VaR. Returns (LR statistic, p-value, n_exceptions, n_expected).
    """
    n = len(exceptions)
    x = int(exceptions.sum())
    n_expected = p * n

    if x == 0:
        # limit case: log(0^0) term drops out
        log_num = n * np.log(1 - p)
        log_den = n * np.log(1 - x / n) if x != n else 0.0
    else:
        phat = x / n
        log_num = (n - x) * np.log(1 - p) + x * np.log(p)
        log_den = (n - x) * np.log(1 - phat) + x * np.log(phat)

    lr = -2 * (log_num - log_den)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_value, x, n_expected


def christoffersen_independence_test(exceptions):
    """Likelihood-ratio test for independence of exceptions (i.e. that
    breaches aren't clustered in time). Returns (LR statistic, p-value).
    """
    I = exceptions.astype(int)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(I)):
        prev, cur = I[t - 1], I[t]
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    log_num = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    log_den = (n00 * safe_log(1 - pi01) + n01 * safe_log(pi01)
               + n10 * safe_log(1 - pi11) + n11 * safe_log(pi11))

    lr = -2 * (log_num - log_den)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_value
