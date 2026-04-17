"""End-to-end run of the fixed income factor risk & performance attribution
engine: bootstrap -> NSS fit -> PCA -> issuer clustering -> key-rate/spread
duration -> historical VaR/ES with Kupiec/Christoffersen backtests ->
Campisi attribution. Prints a summary report and saves a few charts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fire import curve, portfolio, clustering, duration, risk, attribution

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    rng_seed = 42

    # ---- 1. Bootstrap the Treasury zero curve ----------------------------
    key_tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    key_yields = np.array([0.0510, 0.0505, 0.0490, 0.0460, 0.0440,
                            0.0425, 0.0430, 0.0440, 0.0470, 0.0460])
    grid, par_curve = curve.interpolate_par_curve(key_tenors, key_yields)
    zero_rates, disc_factors = curve.bootstrap_zero_curve(grid, par_curve)

    print("=" * 70)
    print("1. TREASURY CURVE BOOTSTRAP")
    print("=" * 70)
    for t in [0.5, 1, 2, 5, 10, 30]:
        idx = list(grid).index(t)
        print(f"  {t:>5.2f}y  par={par_curve[idx]*100:5.2f}%  zero={zero_rates[idx]*100:5.2f}%")

    # ---- 2. NSS fit, multi-start ------------------------------------------
    beta, sse, n_starts = curve.fit_nss_multistart(grid, zero_rates)
    fitted = curve.nss_rate(grid, beta)
    max_err_bps = np.max(np.abs(fitted - zero_rates)) * 1e4

    print("\n" + "=" * 70)
    print("2. NELSON-SIEGEL-SVENSSON FIT")
    print("=" * 70)
    print(f"  starts tried: {n_starts} (8x8 lambda grid)")
    print(f"  beta0..beta3, lambda1, lambda2 = {np.round(beta, 4)}")
    print(f"  max abs fit error: {max_err_bps:.2f} bp")

    # ---- 3. Historical curve simulation + PCA -----------------------------
    curve_changes, scenario_tenors = curve.simulate_historical_curves(
        risk.SCENARIO_TENORS, zero_rates, n_days=1000, seed=rng_seed)
    explained, components = curve.pca_factors(curve_changes)

    print("\n" + "=" * 70)
    print("3. PCA FACTOR RECOVERY (from simulated daily curve moves)")
    print("=" * 70)
    labels3 = ["PC1 (level)", "PC2 (slope)", "PC3 (curvature)"]
    for lbl, var in zip(labels3, explained):
        print(f"  {lbl:<16} explains {var*100:5.2f}% of variance")
    print(f"  top-3 cumulative: {explained.sum()*100:.2f}%")

    # ---- 4. Issuers, portfolio, clustering ---------------------------------
    issuers = portfolio.generate_issuers(n_issuers=120, seed=1)
    bonds = portfolio.generate_portfolio(issuers, key_tenors, key_yields, seed=2)
    issuer_spreads = {i["issuer_id"]: i["spread_level"] for i in issuers}

    n_clusters = 6
    labels, km = clustering.cluster_issuers(issuers, n_clusters=n_clusters, seed=3)

    print("\n" + "=" * 70)
    print("4. ISSUER PEER GROUPS (k-means, k={})".format(n_clusters))
    print("=" * 70)
    for c in range(n_clusters):
        members = [i for i in issuers if i["cluster"] == c]
        avg_spread = np.mean([m["spread_level"] for m in members]) * 1e4
        print(f"  cluster {c}: {len(members):3d} issuers, avg spread {avg_spread:6.1f}bp")

    # ---- 5. Key-rate + spread duration -------------------------------------
    agg_krd, agg_sdur, total_mv = duration.portfolio_durations(beta, issuer_spreads, bonds)

    print("\n" + "=" * 70)
    print("5. PORTFOLIO KEY-RATE & SPREAD DURATION")
    print("=" * 70)
    print(f"  portfolio market value: {total_mv:,.0f}")
    for t, krd in agg_krd.items():
        print(f"  KRD[{t:>2}y] = {krd:6.3f}")
    print(f"  spread duration = {agg_sdur:6.3f}")

    # ---- 6. Historical VaR / ES, full revaluation --------------------------
    base_value = portfolio.portfolio_value(beta, issuer_spreads, bonds)
    issuer_ids = [i["issuer_id"] for i in issuers]
    spread_changes = risk.simulate_spread_changes(issuers, n_days=1000, seed=rng_seed + 1)

    pnl = risk.scenario_pnl(beta, issuer_spreads, bonds, base_value,
                             curve_changes, spread_changes, issuer_ids)
    var_99, es_99 = risk.historical_var_es(pnl, confidence=0.99)

    print("\n" + "=" * 70)
    print("6. 1-DAY 99% HISTORICAL VaR / ES (full revaluation, 1,000 scenarios)")
    print("=" * 70)
    print(f"  VaR(99%) = {var_99:,.0f}  ({var_99/total_mv*1e4:.1f} bp of MV)")
    print(f"  ES(99%)  = {es_99:,.0f}  ({es_99/total_mv*1e4:.1f} bp of MV)")

    # ---- 7. Backtest: Kupiec + Christoffersen ------------------------------
    n_backtest = 1000
    bt_curve_changes, _ = curve.simulate_historical_curves(
        risk.SCENARIO_TENORS, zero_rates, n_days=n_backtest, seed=rng_seed + 2)
    bt_spread_changes = risk.simulate_spread_changes(issuers, n_days=n_backtest, seed=rng_seed + 3)
    bt_pnl = risk.scenario_pnl(beta, issuer_spreads, bonds, base_value,
                                bt_curve_changes, bt_spread_changes, issuer_ids)

    exceptions = (-bt_pnl) > var_99
    lr_uc, p_uc, n_exc, n_exp = risk.kupiec_pof_test(exceptions, p=0.01)
    lr_ind, p_ind = risk.christoffersen_independence_test(exceptions)

    print("\n" + "=" * 70)
    print(f"7. VaR BACKTEST ({n_backtest} days, fixed VaR threshold)")
    print("=" * 70)
    print(f"  exceptions: {n_exc} observed vs {n_exp:.1f} expected")
    print(f"  Kupiec POF:           LR = {lr_uc:6.3f}   p-value = {p_uc:.3f}"
          f"   {'(fail to reject)' if p_uc > 0.05 else '(REJECT calibration)'}")
    print(f"  Christoffersen indep: LR = {lr_ind:6.3f}   p-value = {p_ind:.3f}"
          f"   {'(fail to reject)' if p_ind > 0.05 else '(REJECT independence)'}")

    # ---- 8. Campisi attribution --------------------------------------------
    dt = 21 / 252  # ~1 trading month
    n_days_period = 21
    period_curve_changes, _ = curve.simulate_historical_curves(
        risk.SCENARIO_TENORS, zero_rates, n_days=n_days_period, seed=rng_seed + 4)
    period_spread_changes = risk.simulate_spread_changes(issuers, n_days=n_days_period, seed=rng_seed + 5)

    cum_curve_delta = period_curve_changes.sum(axis=0)
    cum_spread_delta = period_spread_changes.sum(axis=0)

    shifted_grid_rates = curve.nss_rate(grid, beta) + np.interp(grid, risk.SCENARIO_TENORS, cum_curve_delta)
    beta_end, _, _ = curve.fit_nss_multistart(grid, shifted_grid_rates)
    issuer_spreads_end = {
        iid: issuer_spreads[iid] + cum_spread_delta[j] for j, iid in enumerate(issuer_ids)
    }

    agg_attr, attr_mv = attribution.portfolio_campisi_attribution(
        bonds, issuer_spreads, issuer_spreads_end, beta, beta_end, dt)

    print("\n" + "=" * 70)
    print("8. CAMPISI PERFORMANCE ATTRIBUTION (portfolio, 1-month holding period)")
    print("=" * 70)
    for k in ["carry", "roll_down", "curve_shift", "spread_change", "residual", "total_return"]:
        print(f"  {k:<14} {agg_attr[k]*1e4:7.1f} bp")

    # ---- charts -------------------------------------------------------------
    _make_charts(grid, zero_rates, fitted, curve_changes, explained, components,
                 issuers, agg_krd, pnl, var_99, es_99, agg_attr)

    print(f"\nCharts saved to {OUT_DIR}/")


def _make_charts(grid, zero_rates, fitted, curve_changes, explained, components,
                  issuers, agg_krd, pnl, var_99, es_99, agg_attr):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(grid, zero_rates * 100, "o", ms=3, label="bootstrapped")
    ax.plot(grid, fitted * 100, "-", label="NSS fit")
    ax.set_title("Zero curve: bootstrap vs NSS fit")
    ax.set_xlabel("maturity (y)"); ax.set_ylabel("zero rate (%)"); ax.legend()

    ax = axes[0, 1]
    ax.bar(["PC1", "PC2", "PC3"], explained * 100)
    ax.set_title("PCA explained variance")
    ax.set_ylabel("%")

    ax = axes[0, 2]
    spreads_bp = [i["spread_level"] * 1e4 for i in issuers]
    vols_bp = [i["spread_vol"] * 1e4 for i in issuers]
    clusters = [i["cluster"] for i in issuers]
    sc = ax.scatter(spreads_bp, vols_bp, c=clusters, cmap="tab10", s=20)
    ax.set_title("Issuer peer groups (k-means)")
    ax.set_xlabel("spread level (bp)"); ax.set_ylabel("spread vol (bp/day)")

    ax = axes[1, 0]
    tenors = list(agg_krd.keys())
    ax.bar([str(t) for t in tenors], list(agg_krd.values()))
    ax.set_title("Portfolio key-rate duration")
    ax.set_xlabel("key-rate tenor (y)"); ax.set_ylabel("KRD")

    ax = axes[1, 1]
    ax.hist(-pnl, bins=40, color="steelblue", alpha=0.8)
    ax.axvline(var_99, color="crimson", linestyle="--", label=f"VaR 99% = {var_99:,.0f}")
    ax.axvline(es_99, color="darkred", linestyle=":", label=f"ES 99% = {es_99:,.0f}")
    ax.set_title("1-day loss distribution (1,000 scenarios)")
    ax.set_xlabel("loss"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    keys = ["carry", "roll_down", "curve_shift", "spread_change", "residual"]
    vals = [agg_attr[k] * 1e4 for k in keys]
    colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]
    ax.bar(keys, vals, color=colors)
    ax.set_title("Campisi attribution (bp, 1-month)")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "report.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
