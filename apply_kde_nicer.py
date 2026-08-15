#!/usr/bin/env python3
"""
apply_kde_nicer.py
==================

Re-reweight a saved EOS posterior with the tier-2 NICER KDE
likelihood for J0740 and J0437, while keeping J0030 on the
existing summary-Gaussian path.

This script does NOT re-run TOV: it relies on the cached (N,
n_central) M / R / Lambda arrays in
out["reweighting"] from a previous run of run_sampling.py.
The GW, M_max, and (if present) pQCD log-L arrays are also reused
from the cache.

Edit the configuration blocks below and run from the project root:

    python apply_kde_nicer.py

Output:
    NEW_PT (default analysis/res_finale_baseline/eos_EFT_posterior_kdenicer.pt) with:
      reweighting["log_L_nicer"]          combined NICER log-L
                                          (summary J0030 + KDE J0740/J0437)
      reweighting["log_L_nicer_summary"]  summary-only contribution
      reweighting["log_L_nicer_kde"]      KDE-only contribution
      reweighting["weights"]              re-derived joint weights
      reweighting["log_weights"]          re-derived joint log-weights
      reweighting["ESS"]                  re-derived joint ESS
      reweighting["ESS_nicer"]            full NICER ESS
      reweighting["ESS_nicer_summary"]    summary-only ESS
      reweighting["ESS_nicer_kde"]        KDE-only ESS
      reweighting["clamp_counts"]         per-pulsar -1e6 floor counts
                                          (under the new mixed setup)
      weighted_mean / weighted_std / weighted_q16 / weighted_q84
                                          re-derived from new weights
      M / R / Lambda / log_L_gw / log_L_mmax / log_L_pqcd / pqcd_*
                                          PASS-THROUGH from OLD_PT

The script prints a per-term ESS comparison vs. the original .pt so
the user can see how the KDE path affects each individual data
constraint.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

# Allow running this script from the project root without installing
# the eos_sampling package -- the imports below resolve relative.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eos_sampling.reweighting import (
    LIKELIHOOD_VERSION,
    _ess_from_logw,
    _nicer_kde_term,
    _nicer_term_line_integral,
    _stable_branch,
    prepare_nicer_kde_pulsars,
    weighted_summary,
)
from eos_sampling.astro_configs import (NICER_PULSARS_SUMMARY,
                                        NICER_PULSARS_KDE,
                                        NICER_PULSARS_KDE_MILLER)

OLD_PT = "analysis/res_finale_baseline/eos_EFT_posterior_L500.pt"

# Which J0437 NICER analysis feeds the KDE likelihood.  Both variants
# reuse the SAME cached M/R/Lambda and GW/M_max/pQCD log-L from OLD_PT
# (no re-sampling, no TOV) -- only the NICER term is recomputed -- so
# flipping this string and rerunning produces a clean A/B comparison.
#   "choudhury" : Amsterdam analysis (Choudhury+24), the baseline/headline.
#   "miller"    : independent analysis (Miller+25), robustness variant.
#   "summary"   : all three pulsars as summary Gaussians, i.e. exactly
#                 the tier-1 NICER term that run_sampling.py
#                 computes.  Recomputing it here from the cached TOV
#                 arrays is a consistency check on the two code paths:
#                 when OLD_PT carries the current likelihood_version the
#                 result must reproduce it to floating precision, and the
#                 script reports whether it does.  When OLD_PT predates
#                 the current version, this mode re-derives the baseline
#                 under the current likelihood without paying for a full
#                 rerun (see the end notes).
J0437_ANALYSIS = "choudhury"      # "summary", "choudhury" or "miller"

if J0437_ANALYSIS == "summary":
    NICER_PULSARS = NICER_PULSARS_SUMMARY
    NEW_PT = "analysis/res_finale_baseline/eos_EFT_posterior_L500_check.pt"
elif J0437_ANALYSIS == "choudhury":
    NICER_PULSARS = NICER_PULSARS_KDE
    NEW_PT = "analysis/res_finale_baseline/eos_EFT_posterior_kdenicer.pt"
elif J0437_ANALYSIS == "miller":
    NICER_PULSARS = NICER_PULSARS_KDE_MILLER
    NEW_PT = "analysis/res_finale_baseline/eos_EFT_posterior_kdenicer_miller.pt"
else:
    raise ValueError(f"unknown J0437_ANALYSIS: {J0437_ANALYSIS!r} "
                     f"(expected 'summary', 'choudhury' or 'miller')")

def main():
    if not os.path.exists(OLD_PT):
        raise FileNotFoundError(
            f"OLD_PT={OLD_PT!r} not found.  Run run_sampling.py "
            f"first to produce a baseline .pt.")

    print("=" * 64)
    print(f"  apply_kde_nicer: {OLD_PT}")
    print(f"             ->   {NEW_PT}")
    print("=" * 64)

    out = torch.load(OLD_PT, map_location="cpu", weights_only=False)
    r   = out["reweighting"]

    old_version = r.get("likelihood_version", 1)
    print(f"  likelihood version: OLD_PT = {old_version}, "
          f"code = {LIKELIHOOD_VERSION}"
          + ("   [match]" if old_version == LIKELIHOOD_VERSION else ""))
    if old_version != LIKELIHOOD_VERSION:
        sys.exit(
            f"\n  {OLD_PT}\n"
            f"  carries likelihood_version {old_version}; this code is at "
            f"{LIKELIHOOD_VERSION}.  REFUSING to re-reweight it.\n"
            f"\n"
            f"  This script only recomputes the NICER term.  Everything "
            f"else -- M, R, Lambda and the log-L terms formed from them -- "
            f"is carried through from OLD_PT verbatim.  So it can migrate a "
            f"file across a change to the NICER LIKELIHOOD FORM, but not "
            f"across a change to the TOV ARRAYS THEMSELVES, and version 3 "
            f"contains both: the KDE prior divide-out and the mass-prior "
            f"normalisation are recomputable here, the SLy crust and the "
            f"truncation policy are not.\n"
            f"\n"
            f"  Version 2 is the dangerous case specifically.  The crust "
            f"landed BEFORE the version bump, so a version-2 file may be "
            f"crusted or crustless with nothing in it to say which, and "
            f"re-stamping it as {LIKELIHOOD_VERSION} would let a crustless "
            f"cache (R(1.4) low by ~0.75-1.1 km, Lambda(1.4) low by ~60%) "
            f"through every downstream guard.\n"
            f"\n"
            f"  Re-run run_sampling.py to regenerate the baseline, "
            f"then run this script on it.")

    M_arr = np.asarray(r["M"],      dtype=np.float64)
    R_arr = np.asarray(r["R"],      dtype=np.float64)
    L_arr = np.asarray(r["Lambda"], dtype=np.float64)
    if M_arr.ndim != 2:
        raise ValueError(
            f"reweighting['M'] expected (N, n_central); "
            f"got shape {M_arr.shape}")

    N, n_central = M_arr.shape
    print(f"  loaded cache: N={N}, n_central={n_central}")
    print(f"  passing through log_L_gw, log_L_mmax"
          + (", log_L_pqcd" if "log_L_pqcd" in r else "")
          + " from OLD_PT")

    # --- Build KDE objects once (parent-process cache) ---
    print("\n  preparing pulsars...")
    t0 = time.time()
    prepared = prepare_nicer_kde_pulsars(NICER_PULSARS, verbose=True)
    summary_pulsars = [p for p in prepared
                       if p.get("mode", "summary_gaussian") == "summary_gaussian"]
    kde_pulsars     = [p for p in prepared
                       if p.get("mode", "summary_gaussian") == "kde"]
    for p in summary_pulsars:
        print(f"  [summary] {p['name']}: "
              f"(M, sM, R, sR) = ({p['M_obs']}, {p['sigma_M']}, "
              f"{p['R_obs']}, {p['sigma_R']})")
    print(f"  KDE prep done in {time.time() - t0:.2f} s")

    # --- Per-sample log-L over the cached stable branches ---
    print(f"\n  computing per-sample log-L over {N} cached samples...")
    t0 = time.time()
    log_L_nicer_summary = np.full(N, -np.inf, dtype=np.float64)
    log_L_nicer_kde     = np.full(N, -np.inf, dtype=np.float64)
    clamp_counts = {}
    n_no_branch  = 0

    step = max(1, min(200, N // 20))
    for n in range(N):
        M_s, R_s, _ = _stable_branch(M_arr[n], R_arr[n], L_arr[n])
        if M_s.size < 2:
            # No stable branch -- floor on every pulsar.
            n_no_branch += 1
            for p in summary_pulsars + kde_pulsars:
                pname = p.get("name", "pulsar_?")
                clamp_counts[pname] = clamp_counts.get(pname, 0) + 1
            log_L_nicer_summary[n] = (-1.0e6 * len(summary_pulsars)
                                      if summary_pulsars else 0.0)
            log_L_nicer_kde[n]     = (-1.0e6 * len(kde_pulsars)
                                      if kde_pulsars else 0.0)
            if (n + 1) % step == 0 or (n + 1) == N:
                print(f"    {n+1:>6d}/{N}", end="\r", flush=True)
            continue

        lL_s = 0.0
        if summary_pulsars:
            lL_s, ch_s = _nicer_term_line_integral(M_s, R_s, summary_pulsars)
            for k, v in ch_s.items():
                clamp_counts[k] = clamp_counts.get(k, 0) + int(v)
        log_L_nicer_summary[n] = lL_s

        lL_k = 0.0
        if kde_pulsars:
            lL_k, ch_k = _nicer_kde_term(M_s, R_s, kde_pulsars)
            for k, v in ch_k.items():
                clamp_counts[k] = clamp_counts.get(k, 0) + int(v)
        log_L_nicer_kde[n] = lL_k

        if (n + 1) % step == 0 or (n + 1) == N:
            print(f"    {n+1:>6d}/{N}", end="\r", flush=True)
    print()
    print(f"  per-sample loop done in {time.time() - t0:.2f} s")
    if n_no_branch:
        print(f"  WARNING: {n_no_branch}/{N} samples had < 2 stable-branch "
              f"points (NICER terms floored on those samples)")

    if n_no_branch == N:
        raise RuntimeError(
            f"all {N} cached samples have no stable branch: OLD_PT carries "
            f"an empty M/R/Lambda cache.  Re-run the base sampling step "
            f"before this one.")

    log_L_nicer = log_L_nicer_summary + log_L_nicer_kde

    # Truncated samples were rejected upstream; mask explicitly on the
    # flag the base run recorded (gw = None / mmax = None configurations
    # would otherwise let them back in).
    truncated = np.asarray(r.get("truncated",
                                 np.zeros(N, dtype=bool)), dtype=bool)
    n_trunc = int(truncated.sum())
    log_L_nicer_premask = log_L_nicer.copy()
    if n_trunc:
        log_L_nicer[truncated]         = -np.inf
        log_L_nicer_summary[truncated] = -np.inf
        log_L_nicer_kde[truncated]     = -np.inf
        print(f"  {n_trunc}/{N} samples flagged truncated upstream "
              f"({100.0 * n_trunc / N:.2f}%); rejected here too")

    # --- Combine with cached non-NICER terms ---
    log_L_gw   = np.asarray(r["log_L_gw"],   dtype=np.float64)
    log_L_mmax = np.asarray(r["log_L_mmax"], dtype=np.float64)
    _pm = r.get("premask_terms", None)
    if _pm is not None:
        log_L_gw_premask   = np.asarray(_pm["log_L_gw"],   dtype=np.float64)
        log_L_mmax_premask = np.asarray(_pm["log_L_mmax"], dtype=np.float64)
    else:
        log_L_gw_premask   = None
        log_L_mmax_premask = None
    log_L_total_new = log_L_nicer + log_L_gw + log_L_mmax
    pqcd_present = ("log_L_pqcd" in r)
    if pqcd_present:
        log_L_pqcd = np.asarray(r["log_L_pqcd"], dtype=np.float64)
        # safeguard: -inf is the canonical out-of-window value;
        # adding it propagates to log_L_total -> weight 0, which is
        # correct.  Just make sure NaN doesn't sneak in.
        log_L_pqcd = np.where(np.isnan(log_L_pqcd), -np.inf, log_L_pqcd)
        log_L_pqcd_premask = (np.asarray(_pm["log_L_pqcd"], dtype=np.float64)
                              if (_pm is not None and "log_L_pqcd" in _pm)
                              else None)
        log_L_total_new = log_L_total_new + log_L_pqcd

    # --- Truncation cost, measured rather than assumed ---
    _have_premask = (log_L_gw_premask is not None
                     and log_L_mmax_premask is not None
                     and (not pqcd_present or log_L_pqcd_premask is not None))
    log_L_total_premask = None
    trunc_weight_frac = float("nan")
    if _have_premask:
        log_L_total_premask = (log_L_nicer_premask + log_L_gw_premask
                               + log_L_mmax_premask)
        if pqcd_present:
            log_L_total_premask = log_L_total_premask + log_L_pqcd_premask
        if n_trunc and np.any(np.isfinite(log_L_total_premask)):
            _w_pre, _ = _ess_from_logw(log_L_total_premask, N,
                                       context="pre-truncation joint")
            trunc_weight_frac = float(_w_pre[truncated].sum())
            print(f"  truncated samples would have carried "
                  f"{100.0 * trunc_weight_frac:.3f}% of the posterior weight "
                  f"(report this alongside the "
                  f"{100.0 * n_trunc / N:.2f}% count)")
        elif not n_trunc:
            trunc_weight_frac = 0.0
    elif n_trunc:
        print("  NOTE: OLD_PT predates the pre-mask export, so the weight "
              "cost of the truncation policy cannot be measured from it.  "
              "Re-run the base step to get that number.")

    # --- Joint and per-term ESS ---
    weights, ESS_total       = _ess_from_logw(log_L_total_new,     N,
                                              context="joint")
    _,       ESS_nicer       = _ess_from_logw(log_L_nicer,         N,
                                              context="NICER")
    _,       ESS_nicer_summary = _ess_from_logw(log_L_nicer_summary, N,
                                                context="NICER summary")
    _,       ESS_nicer_kde   = _ess_from_logw(log_L_nicer_kde,     N,
                                              context="NICER KDE")
    _,       ESS_gw          = _ess_from_logw(log_L_gw,            N,
                                              context="GW")
    _,       ESS_mmax        = _ess_from_logw(log_L_mmax,          N,
                                              context="M_max")
    if pqcd_present:
        _,   ESS_pqcd        = _ess_from_logw(log_L_pqcd,          N,
                                              context="pQCD")

    # --- Re-derive weighted summary statistics ---
    samples_phys = out.get("samples_phys", None)
    if samples_phys is None:
        raise RuntimeError(
            "OLD_PT does not contain 'samples_phys'; cannot recompute "
            "weighted summary curves.")
    samples_phys_np = (samples_phys.detach().cpu().numpy()
                       if hasattr(samples_phys, "detach") else
                       np.asarray(samples_phys))
    weighted_mean, weighted_std, weighted_q16, weighted_q84 = \
        weighted_summary(samples_phys_np, weights)

    # --- Compose the new output dict (in-place style on a copy) ---
    new_r = dict(r)
    new_r["log_L_nicer"]          = log_L_nicer
    new_r["log_L_nicer_summary"]  = log_L_nicer_summary
    new_r["log_L_nicer_kde"]      = log_L_nicer_kde
    new_r["log_L_exact"]          = log_L_total_new
    new_r["log_weights"]          = log_L_total_new
    new_r["weights"]              = weights
    new_r["ESS"]                  = float(ESS_total)
    new_r["ESS_nicer"]            = float(ESS_nicer)
    new_r["ESS_nicer_summary"]    = float(ESS_nicer_summary)
    new_r["ESS_nicer_kde"]        = float(ESS_nicer_kde)
    new_r["ESS_gw"]               = float(ESS_gw)
    new_r["ESS_mmax"]             = float(ESS_mmax)
    new_r["clamp_counts"]         = clamp_counts
    new_r["likelihood_version"]   = LIKELIHOOD_VERSION
    new_r["likelihood_version_source"] = int(old_version)
    if log_L_total_premask is not None:
        new_r["log_L_exact_premask"] = log_L_total_premask
        new_r["premask_terms"] = {
            "log_L_nicer": log_L_nicer_premask,
            "log_L_gw":    log_L_gw_premask,
            "log_L_mmax":  log_L_mmax_premask,
            **({"log_L_pqcd": log_L_pqcd_premask} if pqcd_present else {}),
        }
    new_r["truncated_weight_fraction"] = float(trunc_weight_frac)
    if pqcd_present:
        new_r["ESS_pqcd"]         = float(ESS_pqcd)

    out_new = dict(out)
    out_new["reweighting"]    = new_r
    out_new["weighted_mean"]  = weighted_mean
    out_new["weighted_std"]   = weighted_std
    out_new["weighted_q16"]   = weighted_q16
    out_new["weighted_q84"]   = weighted_q84
    out_new["nicer_pulsars_kdenicer"] = NICER_PULSARS

    # --- summary mode: consistency check against OLD_PT ---
    if J0437_ANALYSIS == "summary" and "log_L_nicer" in r:
        old_nicer = np.asarray(r["log_L_nicer"], dtype=np.float64)
        finite    = np.isfinite(old_nicer) & np.isfinite(log_L_nicer)
        if finite.any():
            d = (log_L_nicer - old_nicer)[finite]
            spread = float(np.abs(d - d.mean()).max())
            print(f"\n  tier-1 NICER term vs OLD_PT: mean shift "
                  f"{d.mean():+.4f} nats, max deviation about the mean "
                  f"{spread:.2e} nats")
            if spread < 1e-9:
                print("  -> regression check PASSED: the cache path here "
                      "reproduces the pipeline exactly.")
            else:
                print("  -> regression check FAILED: same likelihood "
                      "version but different numbers.  The cache path "
                      "and importance_weights() have drifted apart.")

    out_dir = os.path.dirname(NEW_PT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out_new, NEW_PT)
    print(f"\n  results saved to {NEW_PT}")

    # --- Per-term ESS comparison vs OLD_PT ---
    print("\n" + "=" * 64)
    print("  per-term ESS comparison")
    print("=" * 64)
    print(f"  {'term':<18s}  {'OLD_PT':>10s}  {'NEW_PT':>10s}  {'delta':>10s}")
    print(f"  {'-' * 56}")

    def _line(label, old_v, new_v):
        if old_v is None:
            old_str = "n/a"
        else:
            old_str = f"{old_v:.1f}"
        delta = "" if old_v is None else f"{new_v - old_v:+.1f}"
        print(f"  {label:<18s}  {old_str:>10s}  {new_v:10.1f}  {delta:>10s}")

    _line("ESS (joint)",         r.get("ESS"),                 ESS_total)
    _line("ESS_nicer (total)",   r.get("ESS_nicer"),           ESS_nicer)
    _line("  -- summary",        r.get("ESS_nicer_summary"),   ESS_nicer_summary)
    _line("  -- KDE",            r.get("ESS_nicer_kde"),       ESS_nicer_kde)
    _line("ESS_gw",              r.get("ESS_gw"),              ESS_gw)
    _line("ESS_mmax",            r.get("ESS_mmax"),            ESS_mmax)
    if pqcd_present:
        _line("ESS_pqcd",        r.get("ESS_pqcd"),            ESS_pqcd)

    # NICER clamp report under the new mixed setup
    if clamp_counts:
        total_clamp = sum(clamp_counts.values())
        if total_clamp > 0:
            print("\n  pulsar clamp summary (samples hitting -1e6 floor "
                  "under the NEW mixed setup):")
            for pname, cnt in clamp_counts.items():
                pct = 100.0 * cnt / N
                marker = ("   <-- effectively uninformative"
                          if pct > 50 else "")
                print(f"    {pname}: {cnt}/{N}  ({pct:.1f}%){marker}")

    print("\n  done.")
if __name__ == "__main__":
    main()