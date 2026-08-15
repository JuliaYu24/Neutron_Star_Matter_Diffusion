"""
apply_heavy_mass_constraint.py
==============================================================================
Add a heavy maximum-mass constraint (e.g. PSR J0952-0607, 2.35 +/- 0.17 Msun)
to an EXISTING posterior by post-hoc importance reweighting -- NO new diffusion
sampling, NO new TOV solves, NO pQCD recomputation.

---------------------------
The astro+pQCD layer is self-normalised importance reweighting on a fixed set
of chi-EFT-inpainted samples.  Each sample's maximum mass M_max is already
cached in the results file.  Adding an independent heavy-mass likelihood is one
extra multiplicative factor on the existing weights:

        w_new_n   proportional to   w_old_n * L_Mmax( M_max^(n) )

So the "new run" is seconds of NumPy on cached arrays.  Runs on a laptop.

What it does
------------
  1. loads an existing posterior .pt              (READ-ONLY: input never edited)
  2. reads the per-sample M_max already cached in it
  3. multiplies the existing weights by the heavy-mass likelihood
  4. renormalises (stable log-sum-exp)
  5. writes a NEW .pt whose out["reweighting"]["weights"] is the reweighted
     version, plus traceability fields, and prints the ESS drop + M_TOV shift
  6. (optional) previews R/Lambda at 1.4 and 2.08 Msun before vs after

Use
---
    python apply_heavy_mass_constraint.py
then in notebook_diagnostics_kde.ipynb set
    OUTPUT_PATH = "<the new file printed at the end>"
and Restart & Run All.  Every plot/table/fiducial then uses the new constraint.
==============================================================================
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from eos_sampling.reweighting import (
    LIKELIHOOD_VERSION,
    per_sample_fiducials,
    weighted_quantile,
    weighted_summary,
)

# ----------------------------------------------------------------------
# CONFIG  --  edit these
# ----------------------------------------------------------------------
IN_PATH  = "analysis/res_finale_baseline/eos_EFT_posterior_kdenicer.pt"   # existing posterior (input)
OUT_PATH = "analysis/res_finale_baseline/eos_EFT_posterior_J0952.pt"           # new file to write (output)

LIKELIHOOD_VERSION_EXPECTED = 3

M_LOW     = 2.35      # heavy-mass central value [Msun]   (J0952-0607)
SIGMA     = 0.17      # 1-sigma uncertainty       [Msun]
ONE_SIDED = True      # True  -> FLOOR: penalise only M_max < M_LOW.
                      #          Physically correct for an M_TOV lower bound,
                      #          and matches how J1614 is treated in the run.
                      # False -> two-sided Gaussian "measurement" centred at
                      #          M_LOW.  For this posterior (almost all mass
                      #          below 2.35) the two give nearly identical
                      #          results -- flip it to check.
SHORT_TAG = "+J0952"  # short label used in the printed tables
SHOW_FIDUCIAL_PREVIEW = True   # quick R/Lambda(1.4, 2.08) before/after readout
# ----------------------------------------------------------------------


def _to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def kish_ess(w):
    w = np.asarray(w, dtype=np.float64)
    s2 = np.sum(w ** 2)
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def get_Mmax(out):
    """Per-sample maximum mass: prefer the reweighter's cached M_max_pred,
    else fall back to nanmax of the per-sample M(P_c) array."""
    rw = out["reweighting"]
    if rw.get("M_max_pred", None) is not None:
        return np.asarray(_to_np(rw["M_max_pred"]), float), "M_max_pred (cached)"
    M = np.asarray(_to_np(rw["M"]), float)
    return np.nanmax(M, axis=1), "nanmax of M(P_c) array"


def fiducial_preview(out, w_old, w_new):
    """Optional: weighted R/Lambda at target masses, old vs new weights.
    The extraction is per_sample_fiducials() from the package, so this
    preview, jackknife_errors.py and the notebook all run the same
    estimator on the same code path."""
    rw = out["reweighting"]
    targets = (1.4, 2.08)
    fid = per_sample_fiducials(
        _to_np(rw["M"]), _to_np(rw["R"]), _to_np(rw["Lambda"]),
        target_masses=targets,
        M_max_pred=rw.get("M_max_pred", None))
    Rt = {t: fid[f"R({t})"]      for t in targets}
    Lt = {t: fid[f"Lambda({t})"] for t in targets}

    print("  --- fiducial preview (weighted median, 68% CI) ---")
    for t in targets:
        for name, arr in ((f"R({t:.2f}) [km]", Rt[t]),
                          (f"Lambda({t:.2f})", Lt[t])):
            o = (weighted_quantile(arr, w_old, .16), weighted_quantile(arr, w_old, .50), weighted_quantile(arr, w_old, .84))
            n_ = (weighted_quantile(arr, w_new, .16), weighted_quantile(arr, w_new, .50), weighted_quantile(arr, w_new, .84))
            print(f"    {name:15s}: now {o[1]:8.2f} [{o[0]:.2f}, {o[2]:.2f}]"
                  f"   ->  {SHORT_TAG} {n_[1]:8.2f} [{n_[0]:.2f}, {n_[2]:.2f}]")


def main():
    if not os.path.exists(IN_PATH):
        sys.exit(f"Input posterior not found: {IN_PATH}\n"
                 f"(run this from the project root, "
                 f"or edit IN_PATH).")

    print(f"Loading {IN_PATH} ...")
    out = torch.load(IN_PATH, weights_only=False)   # input file is never modified
    rw = out["reweighting"]

    _v = rw.get("likelihood_version", 1)
    if _v != LIKELIHOOD_VERSION_EXPECTED:
        sys.exit(f"{IN_PATH} carries likelihood_version {_v}, expected "
                 f"{LIKELIHOOD_VERSION_EXPECTED} (code is at "
                 f"{LIKELIHOOD_VERSION}).  Re-run the base sampling step and "
                 f"apply_kde_nicer.py before adding a heavy-mass constraint.")

    if "heavy_mass_constraint" in out:
        sys.exit(f"{IN_PATH} already carries a heavy-mass constraint "
                 f"({out['heavy_mass_constraint'].get('label', '?')}); "
                 f"refusing to apply a second one on top of it.")

    w_old = np.asarray(_to_np(rw["weights"]), float)
    w_old = w_old / w_old.sum()                      # ensure normalised
    M_max, src = get_Mmax(out)
    N = w_old.size
    if M_max.shape[0] != N:
        sys.exit(f"M_max length {M_max.shape[0]} != weights length {N} -- aborting.")

    ess_old = kish_ess(w_old)
    print(f"  samples              : {N}")
    print(f"  M_max source         : {src}")
    print(f"  ESS (current)        : {ess_old:.1f}")

    # Transparency: samples with no defined M_max get zero weight under a
    # mass constraint (they have no stable maximum mass to satisfy it).
    nan_mask = ~np.isfinite(M_max)
    if nan_mask.any():
        print(f"  note: {int(nan_mask.sum())} samples have no finite M_max; "
              f"they carry {100 * w_old[nan_mask].sum():.2f}% of the current "
              f"weight and are set to 0 under the constraint.")

    # ---- heavy-mass log-likelihood on each sample's M_max ----
    logL = np.full(N, -np.inf)
    fin = np.isfinite(M_max)
    if ONE_SIDED:
        dM = np.minimum(M_max[fin] - M_LOW, 0.0)     # 0 at/above the floor
    else:
        dM = M_max[fin] - M_LOW                       # two-sided
    logL[fin] = -0.5 * (dM / SIGMA) ** 2

    # ---- combine with existing weights (independent factor) ----
    # w_new proportional to w_old * exp(logL); stabilise the exponent.
    # Start from the cached LOG weights when available: a sample whose
    # weight underflowed to exactly zero would be permanently dead under
    # log(w_old) even if the new constraint would have revived it.
    if rw.get("log_weights", None) is not None:
        logw_old = np.asarray(_to_np(rw["log_weights"]), float)
        src_w = "log_weights (cached, no underflow)"
    else:
        with np.errstate(divide="ignore"):
            logw_old = np.log(np.where(w_old > 0, w_old, np.nan))
        src_w = "log(weights) (log_weights absent from this .pt)"
    print(f"  weight source        : {src_w}")

    logw = logw_old + logL
    if not np.any(np.isfinite(logw)):
        sys.exit("no sample has a finite weight under the heavy-mass "
                 "constraint; the constraint is incompatible with this "
                 "posterior and no reweighting is defined.")
    logw_shift = logw - np.nanmax(logw[np.isfinite(logw)])
    w_new = np.where(np.isfinite(logw_shift), np.exp(logw_shift), 0.0)
    w_new = w_new / w_new.sum()
    ess_new = kish_ess(w_new)

    print(f"  ESS ({SHORT_TAG} {M_LOW}+/-{SIGMA}) : {ess_new:.1f}"
          f"   ({100 * ess_new / ess_old:.0f}% of current)")

    # ---- M_TOV shift ----
    print(f"  M_TOV [Msun]         :     q16     q50     q84")
    print(f"    current            :  "
          f"{weighted_quantile(M_max, w_old, .16):6.3f}  "
          f"{weighted_quantile(M_max, w_old, .50):6.3f}  "
          f"{weighted_quantile(M_max, w_old, .84):6.3f}")
    print(f"    {SHORT_TAG:<18s} :  "
          f"{weighted_quantile(M_max, w_new, .16):6.3f}  "
          f"{weighted_quantile(M_max, w_new, .50):6.3f}  "
          f"{weighted_quantile(M_max, w_new, .84):6.3f}")

    if SHOW_FIDUCIAL_PREVIEW:
        fiducial_preview(out, w_old, w_new)

    # ---- re-derive the weighted summary curves ----
    # These top-level keys are what the band plots read; updating
    # rw["weights"] without them would ship a pre-constraint band next
    # to post-constraint tables.
    samples_phys = out.get("samples_phys", None)
    if samples_phys is None:
        sys.exit("input .pt does not contain 'samples_phys'; the weighted "
                 "summary curves cannot be recomputed, and shipping the "
                 "stale ones would put a pre-constraint band next to "
                 "post-constraint tables.")
    sp = np.asarray(_to_np(samples_phys), dtype=np.float64)
    wm, ws, q16c, q84c = weighted_summary(sp, w_new)
    out["weighted_mean"] = wm
    out["weighted_std"]  = ws
    out["weighted_q16"]  = q16c
    out["weighted_q84"]  = q84c
    print(f"  weighted band        : recomputed on {sp.shape[0]} curves "
          f"x {sp.shape[1]} grid points")

    # ---- write NEW .pt (original on disk untouched) ----
    rw["weights"]       = w_new          # the field the diagnostic notebook reads
    rw["weights_preJ"]  = w_old          # keep old weights for traceability
    rw["ESS"]           = ess_new        # keep the notebook's ESS printout honest
    rw["log_weights_preJ"] = logw_old
    rw["log_weights"]      = logw
    rw["log_L_exact"]      = logw
    out["heavy_mass_constraint"] = {
        "label":        SHORT_TAG,
        "M_low":        M_LOW,
        "sigma":        SIGMA,
        "one_sided":    ONE_SIDED,
        "M_max_source": src,
        "ESS_before":   ess_old,
        "ESS_after":    ess_new,
        "source_pt":    os.path.abspath(IN_PATH),
    }
    _out_dir = os.path.dirname(OUT_PATH)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    torch.save(out, OUT_PATH)

    print(f"\nWrote {OUT_PATH}")
    print(f"-> In notebook_diagnostics_kde.ipynb set")
    print(f'       OUTPUT_PATH = "{OUT_PATH}"')
    print(f"   then Restart & Run All.  Every plot/table/fiducial will use the "
          f"new constraint.")
    print(f"\n   ESS drop is itself a consistency check: {ess_old:.0f} -> "
          f"{ess_new:.0f} measures how much the heavy mass disagrees with your "
          f"incumbent (soft) data under the prior.")


if __name__ == "__main__":
    main()
