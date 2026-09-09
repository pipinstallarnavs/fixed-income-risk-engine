"""Treasury zero curve: bootstrapping from par yields, Nelson-Siegel-Svensson
fitting with multi-start optimization, and PCA factor recovery from
historical curve moves.
"""
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def interpolate_par_curve(key_tenors, key_yields, grid_step=0.5, max_tenor=30.0):
    """Linearly interpolate par yields given at a few key tenors onto a
    regular semiannual grid, since bootstrapping needs a cashflow at every
    coupon date.
    """
    grid = np.arange(grid_step, max_tenor + 1e-9, grid_step)
    yields = np.interp(grid, key_tenors, key_yields)
    return grid, yields


def bootstrap_zero_curve(tenors, par_yields, freq=2):
    """Standard semiannual bootstrap: par yields -> continuously-compounded
    zero rates. Assumes `tenors` is an evenly spaced 1/freq-year grid
    starting at 1/freq, i.e. every coupon date is present, so the price of
    each successive par bond can be spanned entirely by already-known
    discount factors plus one new one.
    """
    n = len(tenors)
    zero_rates = np.zeros(n)
    disc_factors = np.zeros(n)
    coupon_period = 1.0 / freq

    for i, T in enumerate(tenors):
        c = par_yields[i] / freq * 100.0  # coupon paid per period, face=100

        if i == 0:
            # only one cashflow (first coupon + redemption is still far off);
            # this node is priced like a pure discount instrument
            price = 100.0
            disc_factors[i] = price / (100.0 + c) if T == coupon_period else 1.0
            # first node: 100 = (100+c) * DF  =>  DF = 100/(100+c)
            disc_factors[i] = 100.0 / (100.0 + c)
        else:
            pv_coupons = sum(c * disc_factors[j] for j in range(i))
            disc_factors[i] = (100.0 - pv_coupons) / (100.0 + c)

        zero_rates[i] = -np.log(disc_factors[i]) / T

    return zero_rates, disc_factors


# ---------------------------------------------------------------------------
# Nelson-Siegel-Svensson
# ---------------------------------------------------------------------------

def nss_rate(tau, beta):
    """NSS zero rate at maturity tau (years). beta = [b0, b1, b2, b3, l1, l2]."""
    b0, b1, b2, b3, l1, l2 = beta
    tau = np.asarray(tau, dtype=float)
    # guard against division by zero at tau=0
    x1 = np.where(tau > 1e-8, tau / l1, 1e-8)
    x2 = np.where(tau > 1e-8, tau / l2, 1e-8)
    term1 = (1 - np.exp(-x1)) / x1
    term2 = term1 - np.exp(-x1)
    term3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1 * term1 + b2 * term2 + b3 * term3


def _fit_nss_given_lambdas(tenors, zero_rates, l1, l2):
    """With l1, l2 fixed, the NSS rate is *linear* in b0..b3, so this inner
    fit is just an ordinary least squares regression -> closed form, no
    local minima. This is what makes the multi-start over (l1, l2) tractable:
    we only ever search a 2D non-convex surface, not a 6D one.
    """
    x1 = tenors / l1
    x2 = tenors / l2
    f0 = np.ones_like(tenors)
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    X = np.column_stack([f0, f1, f2, f3])
    beta_linear, *_ = np.linalg.lstsq(X, zero_rates, rcond=None)
    fitted = X @ beta_linear
    sse = np.sum((fitted - zero_rates) ** 2)
    return beta_linear, sse


def fit_nss_multistart(tenors, zero_rates, lambda_grid=None):
    """Fit NSS via a grid of starting points for (lambda1, lambda2), the two
    decay parameters. These enter the model nonlinearly and are notoriously
    unidentifiable from a single start: many (l1, l2) pairs produce nearly
    identical curve shapes, so a local optimizer seeded from one guess
    routinely lands in the wrong basin. We instead grid over a set of
    starting lambdas (8 x 8 = 64 combinations by default), run a bounded
    local optimizer from each, and keep the best fit.

    Returns (beta, sse, n_starts).
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.5, 15, 8)

    best_sse = np.inf
    best_beta = None

    for l1 in lambda_grid:
        for l2 in lambda_grid:
            if abs(l1 - l2) < 1e-6:
                continue  # degenerate, term2/term3 collapse

            def objective(l, tenors=tenors, zero_rates=zero_rates):
                l1_, l2_ = l
                if l1_ <= 0.05 or l2_ <= 0.05:
                    return 1e6
                _, sse = _fit_nss_given_lambdas(tenors, zero_rates, l1_, l2_)
                return sse

            res = minimize(
                objective, x0=[l1, l2], method="L-BFGS-B",
                bounds=[(0.1, 30.0), (0.1, 30.0)],
                options={"maxiter": 200},
            )
            if res.fun < best_sse:
                l1_opt, l2_opt = res.x
                beta_linear, sse = _fit_nss_given_lambdas(tenors, zero_rates, l1_opt, l2_opt)
                if sse < best_sse:
                    best_sse = sse
                    best_beta = np.array([*beta_linear, l1_opt, l2_opt])

    n_starts = len(lambda_grid) ** 2
    return best_beta, best_sse, n_starts


# ---------------------------------------------------------------------------
# Historical curve simulation + PCA factor recovery
# ---------------------------------------------------------------------------

def simulate_historical_curves(key_rate_tenors, base_zero_rates, n_days=750, seed=0):
    """Simulate a daily history of the zero curve at a handful of key-rate
    tenors as level + slope + curvature shocks (a standard 3-factor picture
    of yield curve dynamics) plus a bit of idiosyncratic tenor noise. This
    stands in for a real historical rates database and gives us something
    honest for PCA to recover the factors from, and for VaR to resample.

    Returns a DataFrame-free (n_days, n_tenors) array of *daily rate changes*
    (in decimal, e.g. 0.0001 = 1bp) plus the tenor grid.
    """
    rng = np.random.default_rng(seed)
    tenors = np.asarray(key_rate_tenors, dtype=float)
    n_tenors = len(tenors)

    # basis shapes: level (flat), slope (short vs long), curvature (belly vs wings)
    level = np.ones(n_tenors)
    slope = (tenors - tenors.mean()) / tenors.std()
    curvature = -((tenors - np.median(tenors)) ** 2)
    curvature = (curvature - curvature.mean()) / curvature.std()

    # daily factor shocks, level dominates, slope less, curvature least
    level_shocks = rng.normal(0, 6e-4, n_days)       # ~6bp/day level vol
    slope_shocks = rng.normal(0, 2.5e-4, n_days)      # ~2.5bp/day slope vol
    curv_shocks = rng.normal(0, 1.2e-4, n_days)       # ~1.2bp/day curvature vol
    idio_noise = rng.normal(0, 0.25e-4, (n_days, n_tenors))  # small per-tenor noise

    changes = (
        np.outer(level_shocks, level)
        + np.outer(slope_shocks, slope)
        + np.outer(curv_shocks, curvature)
        + idio_noise
    )
    return changes, tenors


def pca_factors(rate_changes, n_components=3):
    """PCA on daily zero-rate changes at the key-rate tenors. Returns
    (explained_variance_ratio, components) where components[i] is the i-th
    principal axis (should look like level / slope / curvature in order).
    """
    X = rate_changes - rate_changes.mean(axis=0, keepdims=True)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained = eigvals / eigvals.sum()
    return explained[:n_components], eigvecs[:, :n_components].T
