#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eos_sampling.reweighting import (
    LIKELIHOOD_VERSION,
    per_sample_fiducials,
    weighted_quantile,
)


IN_PT  = "analysis/res_finale_baseline/eos_EFT_posterior_kdenicer.pt"

OUT_PT = "analysis/res_finale_baseline/eos_EFT_mc_error.pt"

LIKELIHOOD_VERSION_EXPECTED = 3


TARGET_MASSES = (1.4, 2.08)

N_GROUPS = 100
N_BOOT   = 500
CI       = (16.0, 84.0)
SEED     = 12345

MAXW_WARN   = 0.05
ESSFRAC_WARN = 0.005


def _to_np(x):
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)


def ess_and_maxweight(weights):
    """Kish ESS and the largest single normalized weight."""
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return float(w.size), 1.0 / max(w.size, 1)
    wn = w / s
    ess = 1.0 / np.sum(wn * wn)
    return float(ess), float(wn.max())



def block_jackknife_quantile(values, weights, q=0.50,
                             n_groups=100, seed=0):
    """
    Delete-d (grouped / block) jackknife MC standard error of a
    weighted quantile.  Lattice-style: remove one block, recompute the
    weighted quantile on the remaining samples, repeat over all blocks.

        Var = (G-1)/G * sum_g (theta_(g) - mean_g theta)^2

    with G = number of finite block estimates.  Consistent for
    quantiles because each deleted block is large.

    Returns (point, se, n_groups_used).
    """
    values  = np.asarray(values,  dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    N = values.size
    point = weighted_quantile(values, weights, q)

    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(N)
    blocks = np.array_split(perm, n_groups)

    theta = np.full(len(blocks), np.nan)
    for g, blk in enumerate(blocks):
        keep = np.ones(N, dtype=bool)
        keep[blk] = False
        theta[g] = weighted_quantile(values[keep], weights[keep], q)

    ok = np.isfinite(theta)
    G  = int(ok.sum())
    if G < 2:
        return point, float("nan"), G
    tbar = theta[ok].mean()
    var  = (G - 1) / G * np.sum((theta[ok] - tbar) ** 2)
    return point, float(np.sqrt(var)), G


def bootstrap_quantile(values, weights, q=0.50, n_boot=500,
                       ci=(16.0, 84.0), seed=0):
    """
    Bootstrap MC standard error + MC credible interval of a weighted
    quantile.  Resamples the (value, weight) PAIRS with replacement N
    times; weights are renormalized inside weighted_quantile.

    Returns (point, se, ci_low, ci_high, n_finite_reps).
    """
    values  = np.asarray(values,  dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    N = values.size
    point = weighted_quantile(values, weights, q)

    rng  = np.random.default_rng(seed)
    reps = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        reps[b] = weighted_quantile(values[idx], weights[idx], q)

    ok = np.isfinite(reps)
    if ok.sum() < 2:
        return point, float("nan"), float("nan"), float("nan"), int(ok.sum())
    se = float(np.std(reps[ok], ddof=1))
    lo = float(np.percentile(reps[ok], ci[0]))
    hi = float(np.percentile(reps[ok], ci[1]))
    return point, se, lo, hi, int(ok.sum())


def snis_mean_se(values, weights):
    """
    Closed-form (delta-method) SNIS Monte-Carlo SE of the weighted MEAN,
    evaluated per column:

        Var[mu_hat]_j ~ sum_i wtil_i^2 (f_ij - mu_j)^2 ,  sum_i wtil_i = 1.

    No resampling; O(N*L).  Use for the c_s^2 band mean curve (smooth,
    so no bootstrap needed).  `values` is (N,) or (N, L).
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if v.ndim == 1:
        mu = np.sum(w * v)
        return float(np.sqrt(np.sum(w ** 2 * (v - mu) ** 2)))
    mu = (w[:, None] * v).sum(axis=0)
    return np.sqrt((w[:, None] ** 2 * (v - mu[None, :]) ** 2).sum(axis=0))

def main():
    if not os.path.exists(IN_PT):
        raise FileNotFoundError(
            f"IN_PT={IN_PT!r} not found.  Run run_sampling.py "
            f"(and apply_kde_nicer.py) first to produce a posterior .pt.")

    print("=" * 70)
    print(f"  jackknife_errors:  {IN_PT}")
    print(f"               ->    {OUT_PT}")
    print("=" * 70)

    out = torch.load(IN_PT, map_location="cpu", weights_only=False)
    r   = out["reweighting"]

    v = r.get("likelihood_version", 1)
    if v != LIKELIHOOD_VERSION_EXPECTED:
        raise SystemExit(
            f"{IN_PT} carries likelihood_version {v}, expected "
            f"{LIKELIHOOD_VERSION_EXPECTED} (code is at {LIKELIHOOD_VERSION})."
            f"  Re-run the base sampling step and apply_kde_nicer.py before "
            f"reporting MC errors.")

    M = _to_np(r["M"]).astype(np.float64)
    R = _to_np(r["R"]).astype(np.float64)
    L = _to_np(r["Lambda"]).astype(np.float64)
    w = _to_np(r["weights"]).astype(np.float64)
    if M.ndim != 2:
        raise ValueError(f"reweighting['M'] expected (N, n_central); "
                         f"got {M.shape}")
    N = M.shape[0]

    ess, wmax = ess_and_maxweight(w)
    print(f"\n  N (cached, post-causality) = {N}")
    print(f"  ESS (Kish)                 = {ess:.1f}   "
          f"(ESS/N = {100.0 * ess / N:.2f}%)")
    print(f"  max single normalized weight = {wmax:.4f}")
    if wmax > MAXW_WARN or (ess / N) < ESSFRAC_WARN:
        print("  --------------------------------------------------------")
        print("  WARNING: the weight distribution is concentrated.")
        print("  The bootstrap/jackknife MC bars assume no single sample")
        print("  dominates (finite weight variance).  With this ESS the")
        print("  bars are usable but the honest fix for a small ESS is")
        print("  MORE SAMPLES or a better-aligned proposal, not a fancier")
        print("  estimator.  Treat the MC numbers below as indicative.")
        print("  --------------------------------------------------------")

    print("\n  building per-sample fiducials (stable-branch interpolation)...")
    t0 = time.time()
    per_sample = per_sample_fiducials(
        M, R, L,
        target_masses=TARGET_MASSES,
        M_max_pred=r.get("M_max_pred", None))
    print(f"    done in {time.time() - t0:.2f} s")
    print(f"\n  resampling (jackknife: {N_GROUPS} blocks; "
          f"bootstrap: {N_BOOT} reps)...")
    t0 = time.time()
    results = {}
    print()
    print(f"  {'quantity':<14s} {'median':>9s} {'post 68% CI':>20s} "
          f"{'MC-SE(jk)':>10s} {'MC-SE(bs)':>10s} {'MC 68% CI':>20s} "
          f"{'Nvalid':>7s}")
    print("  " + "-" * 95)
    for name, vals in per_sample.items():
        n_valid = int(np.isfinite(vals).sum())
        med = weighted_quantile(vals, w, 0.50)
        q16 = weighted_quantile(vals, w, 0.16)
        q84 = weighted_quantile(vals, w, 0.84)

        pj, se_j, G          = block_jackknife_quantile(
            vals, w, 0.50, n_groups=N_GROUPS, seed=SEED)
        pb, se_b, lo, hi, nb = bootstrap_quantile(
            vals, w, 0.50, n_boot=N_BOOT, ci=CI, seed=SEED)

        results[name] = {
            "median":          med,
            "post_q16":        q16,
            "post_q84":        q84,
            "n_valid":         n_valid,
            "mc_se_jackknife": se_j,
            "jackknife_blocks": G,
            "mc_se_bootstrap": se_b,
            "mc_ci_low":       lo,
            "mc_ci_high":      hi,
            "n_boot_finite":   nb,
        }
        print(f"  {name:<14s} {med:9.3f} "
              f"[{q16:8.3f},{q84:8.3f}] "
              f"{se_j:10.4f} {se_b:10.4f} "
              f"[{lo:8.3f},{hi:8.3f}] {n_valid:7d}")
    print(f"\n    resampling done in {time.time() - t0:.2f} s")
    band = None
    if "samples_phys" in out and out["samples_phys"] is not None:
        samples = _to_np(out["samples_phys"]).astype(np.float64)
        mc_se_curve = snis_mean_se(samples, w)
        nB_grid     = _to_np(out["nB_grid"]).astype(np.float64) \
            if "nB_grid" in out else None
        post_std    = _to_np(out["weighted_std"]).astype(np.float64) \
            if out.get("weighted_std") is not None else None
        band = {"mc_se_mean_curve": mc_se_curve, "nB_grid": nB_grid}
        msg = f"\n  c_s^2 band mean: max MC-SE = {np.nanmax(mc_se_curve):.3e}"
        if post_std is not None:
            good = post_std > 1e-6
            ratio = np.full_like(mc_se_curve, np.nan)
            ratio[good] = mc_se_curve[good] / post_std[good]
            band["mc_se_over_post_std"] = ratio
            msg += (f"   |   max (MC-SE / posterior-std) = "
                    f"{np.nanmax(ratio):.3f}")
        print(msg)
        print("  (MC-SE / posterior-std << 1 means the band is well "
              "converged at N.)")
    meta = {
        "source_pt":     os.path.abspath(IN_PT),
        "N":             N,
        "ESS":           ess,
        "max_weight":    wmax,
        "target_masses": list(TARGET_MASSES),
        "n_groups":      N_GROUPS,
        "n_boot":        N_BOOT,
        "ci":            list(CI),
        "seed":          SEED,
        "likelihood_version": int(v),
        "note": ("MC error = finite-sample (convergence) error of the "
                 "weighted estimators; NOT the physical uncertainty. "
                 "Physical error bar = weighted posterior 16/84 CI."),
    }
    payload = {"mc_error": {"fiducials": results, "cs2_band": band,
                            "meta": meta}}

    out_dir = os.path.dirname(OUT_PT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, OUT_PT)
    print(f"\n  saved MC-error results to {OUT_PT}")
    print("  (input .pt was NOT modified.)")
    print("\n  REMINDER: quote the posterior 68% CI as the uncertainty on")
    print("  each quantity; quote the MC-SE only as a convergence check.")
    print("\n  done.")
    return payload


if __name__ == "__main__":
    main()