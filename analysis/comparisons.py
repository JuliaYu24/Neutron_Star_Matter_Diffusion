"""
Comparison with the Annala et al. 2023 EOS posterior ensembles.

Two release formats are supported:

  1. GP method ensemble (Zenodo DOI 10.5281/zenodo.10101447):
     a single (df, n) pickle.  Loaded by load_annala_ensemble.

  2. Interpolation ensemble (Zenodo DOI 10.5281/zenodo.10102436):
     a directory of .npy files (cs2.npy, n_long_grid.npy, ...).
     Loaded by load_annala_interpolation_ensemble.

Both feed into compute_external_band / compute_external_M_R_band and
overlay automatically through plot_diagnostics_4panel and plot_M_R_band.

"""

from __future__ import annotations

import os
import pickle

import numpy as np

from .diagnostics import (
    compute_P_eps_ensemble, compute_diagnostics, _quantile_band, _mr_band,
)


# Conversion: Annala stores e, p in GeV / fm^3.
# Our pipeline uses MeV / fm^3 throughout (same as eos_sampling.reweighting).
GEV_TO_MEV = 1000.0


def load_annala_ensemble(path,
                         target_nB_grid=None,
                         apply_published_weights=True,
                         use_published_P_eps=True,
                         eps_ref=None, P_ref=None,
                         verbose=True):
    """
    Load an Annala et al. 2023 ensemble pickle and put everything on
    our n_B / n_0 grid (passed via target_nB_grid).

    Parameters
    ----------
    path : str
        Path to EoS_ensemble.pickle .
    target_nB_grid : (L,) array
        Our n_B / n_0 grid.  Pass out['nB_grid'].
    apply_published_weights : bool
        If True, combine the four constraint columns into one
        per-sample weight w = X_rays * r_J0348 * QCD_10ns * TD_BH.
        If False, return uniform weights (useful for inspecting the
        unweighted prior).
    use_published_P_eps : bool
        If True, use df.p and df.e directly (converted to MeV/fm^3),
        interpolated onto target_nB_grid.  If False, recompute P, eps
        from c_s^2 with our Heun integrator (consistency check).
    verbose : bool
        Print summary info.

    Returns
    -------
    ext : dict with keys
        cs2_arr      (N, L)   c_s^2 on our grid (NaN outside Annala's grid)
        P_arr        (N, L)   pressure in MeV/fm^3 on our grid
        eps_arr      (N, L)   energy density in MeV/fm^3 on our grid
        weights      (N,)     combined posterior weights
        native_grid  (L_n,)   Annala's grid n (n_B / n_s)
        M_arr, R_arr, Lambda_arr  per-sample TOV arrays as stored in df
                                  (kept as object arrays of variable length;
                                  use _stable_branch directly when needed)
        M_TOV        (N,)     scalar TOV mass per sample, copied from df.mmax
        N            int      number of samples
        ref          str      'Annala et al. 2023 (Nat. Comm. 14, 8451)'
    """
    if target_nB_grid is None:
        raise ValueError("target_nB_grid is required (pass out['nB_grid']).")
    target = np.asarray(target_nB_grid, dtype=np.float64)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if verbose:
        print(f"  load_annala_ensemble: opening {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")

    with open(path, "rb") as f:
        df, n_native = pickle.load(f)

    n_native = np.asarray(n_native, dtype=np.float64)
    N = len(df)
    L_n = n_native.size
    L = target.size

    if verbose:
        print(f"    samples: {N}")
        print(f"    native grid: {L_n} points in [{n_native.min():.3f}, "
              f"{n_native.max():.3f}] n_s")
        print(f"    target grid: {L} points in [{target.min():.3f}, "
              f"{target.max():.3f}] n_s")


    cs2_native = np.stack([np.asarray(c, dtype=np.float64) for c in df["cs2"]],
                          axis=0)
    e_native_GeV = np.stack([np.asarray(e, dtype=np.float64) for e in df["e"]],
                            axis=0)
    p_native_GeV = np.stack([np.asarray(p, dtype=np.float64) for p in df["p"]],
                            axis=0)

    for name, arr in [("cs2", cs2_native),
                      ("e",   e_native_GeV),
                      ("p",   p_native_GeV)]:
        if arr.shape[1] != L_n:
            raise ValueError(f"Column '{name}' has length {arr.shape[1]} "
                             f"but native grid length is {L_n}.")


    in_range = (target >= n_native.min()) & (target <= n_native.max())
    n_oor = int((~in_range).sum())
    if verbose and n_oor > 0:
        print(f"    {n_oor}/{L} target points lie outside Annala's grid -- "
              f"those slots will be NaN.")

    cs2_arr = np.full((N, L), np.nan)
    eps_arr = np.full((N, L), np.nan)
    P_arr   = np.full((N, L), np.nan)

    for i in range(N):
        cs2_arr[i, in_range] = np.interp(
            target[in_range], n_native, cs2_native[i])
        eps_arr[i, in_range] = np.interp(
            target[in_range], n_native, e_native_GeV[i]) * GEV_TO_MEV
        P_arr[i, in_range]   = np.interp(
            target[in_range], n_native, p_native_GeV[i]) * GEV_TO_MEV


    # (cross-check; not used by default).
    if not use_published_P_eps:
        if eps_ref is None or P_ref is None:
            raise ValueError(
                "load_annala_ensemble: use_published_P_eps=False re-"
                "integrates the ensemble and therefore needs the same "
                "explicit (eps_ref, P_ref) as the posterior -- pass "
                "out['ref_point'] values.")
        if n_oor > 0:
            raise ValueError("use_published_P_eps=False requires the entire "
                             "target grid to lie inside Annala's grid; "
                             "otherwise the integrator has nothing to start "
                             "from on the missing low-density edge.")
        P_arr, eps_arr = compute_P_eps_ensemble(
            cs2_arr, target, eps_ref=eps_ref, P_ref=P_ref, verbose=False)


    causal = np.full(N, True)
    for i in range(N):
        # Only check the in-range portion; out-of-range NaNs are not a defect.
        c = cs2_arr[i, in_range]
        if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
            causal[i] = False

    n_drop = int((~causal).sum())
    if verbose and n_drop > 0:
        print(f"    dropping {n_drop}/{N} samples that fail causality / "
              f"finite check.")

    cs2_arr = cs2_arr[causal]
    P_arr   = P_arr[causal]
    eps_arr = eps_arr[causal]


    if apply_published_weights:
        for col in ("X_rays", "r_J0348", "QCD_10ns", "TD_BH"):
            if col not in df.columns:
                raise KeyError(f"expected column '{col}' is missing from df. "
                               f"Available: {list(df.columns)}")
        w_full = np.asarray(
            df["X_rays"] * df["r_J0348"] * df["QCD_10ns"] * df["TD_BH"],
            dtype=np.float64)
        weights = w_full[causal]
    else:
        weights = np.ones(int(causal.sum()), dtype=np.float64)


    # Stored per-sample as variable-length arrays; keep as object arrays.
    M_arr   = np.array([np.asarray(m, dtype=np.float64) for m in df["m"]],
                       dtype=object)[causal]
    R_arr   = np.array([np.asarray(r, dtype=np.float64) for r in df["r"]],
                       dtype=object)[causal]
    Lam_arr = np.array([np.asarray(L, dtype=np.float64) for L in df["L"]],
                       dtype=object)[causal]
    if "mmax" in df.columns:
        M_TOV = np.asarray(df["mmax"], dtype=np.float64)[causal]
    else:
        M_TOV = np.array([np.nanmax(m) for m in M_arr], dtype=np.float64)

    if verbose:
        N_kept = cs2_arr.shape[0]
        ess_full = (weights.sum() ** 2) / (weights ** 2).sum() if weights.sum() > 0 else 0
        print(f"    kept {N_kept} samples; effective sample size (Kish) "
              f"under combined weights = {ess_full:.1f}")

    return {
        "cs2_arr":      cs2_arr,
        "P_arr":        P_arr,
        "eps_arr":      eps_arr,
        "weights":      weights,
        "native_grid":  n_native,
        "target_grid":  target,
        "M_arr":        M_arr,
        "R_arr":        R_arr,
        "Lambda_arr":   Lam_arr,
        "M_TOV":        M_TOV,
        "N":            int(cs2_arr.shape[0]),
        "ref":          "Annala et al. 2023 (Nat. Commun. 14, 8451)",
        "in_range":     in_range,
    }


def compute_external_band(ext, q_low=0.16, q_high=0.84,
                          label=None, color="darkorange", verbose=True):
    """
    Convert the dict returned by load_annala_ensemble or
    load_annala_interpolation_ensemble into a band dict suitable for
    the external_bands argument of plot_diagnostics_4panel.

    For each diagnostic (gamma, Delta, d_c), the function uses the
    published per-sample curves when they exist on the loaded ext dict
    (keys 'gamma_published', 'Delta_published', 'dc_published') and
    falls back to recomputing from (P, eps, c_s^2) otherwise.  The cs2
    band always uses the published cs2 array (there is nothing to
    recompute).  When verbose=True, each diagnostic announces which
    path was used.

    NaN slots from out-of-range interpolation are propagated into the
    quantiles (NaN-safe weighted_quantile from eos_sampling.reweighting
    handles them: those grid points show up as gaps in the overlay).

    Returns
    -------
    band : dict with keys 'label', 'color', 'cs2', 'gamma', 'Delta', 'd_c',
           'target_grid', 'N', plus 'sources' explaining where each
           diagnostic came from.  Each diagnostic entry is a (3, L) array
           stacking (q_low, q_median, q_high).
    """
    cs2 = ext["cs2_arr"]
    P   = ext["P_arr"]
    eps = ext["eps_arr"]
    w   = ext["weights"]
    target = ext["target_grid"]

    # Recompute as a fallback so we always have something for each diagnostic.
    gamma_calc, Delta_calc, dc_calc = compute_diagnostics(P, eps, cs2)

    gamma_pub = ext.get("gamma_published")
    Delta_pub = ext.get("Delta_published")
    dc_pub    = ext.get("dc_published")

    sources = {}
    if gamma_pub is not None:
        gamma_arr   = gamma_pub
        sources["gamma"] = "published"
    else:
        gamma_arr   = gamma_calc
        sources["gamma"] = "recomputed"
    if Delta_pub is not None:
        Delta_arr   = Delta_pub
        sources["Delta"] = "published"
    else:
        Delta_arr   = Delta_calc
        sources["Delta"] = "recomputed"
    if dc_pub is not None:
        dc_arr      = dc_pub
        sources["d_c"] = "published"
    else:
        dc_arr      = dc_calc
        sources["d_c"] = "recomputed"
    sources["cs2"] = "published"

    if verbose:
        print(f"  compute_external_band: diagnostic sources for "
              f"'{label or ext.get('ref', 'external')}'")
        for k in ("cs2", "gamma", "Delta", "d_c"):
            print(f"    {k:6s}: {sources[k]}")

    return {
        "label":       label or ext.get("ref", "external"),
        "color":       color,
        "cs2":         _quantile_band(cs2, w, q_low, q_high),
        "gamma":       _quantile_band(gamma_arr, w, q_low, q_high),
        "Delta":       _quantile_band(Delta_arr, w, q_low, q_high),
        "d_c":         _quantile_band(dc_arr, w, q_low, q_high),
        "target_grid": target,
        "N":           int(cs2.shape[0]),
        "sources":     sources,
    }


def compute_external_M_R_band(ext, M_grid=None,
                              q_low=0.16, q_high=0.84,
                              label=None, color="darkorange"):
    """
    Variant of compute_external_band for the M-R panel.  Uses Annala's
    per-sample TOV arrays (df.m, df.r), runs them through
    eos_sampling.reweighting._stable_branch, then takes pointwise
    weighted quantiles on a common M grid.

    Returns
    -------
    band : dict with 'label', 'color', 'R' (shape (3, len(M_grid))),
           'M_grid', 'N'.
    """
    M_arr   = ext["M_arr"]
    R_arr   = ext["R_arr"]
    Lam_arr = ext["Lambda_arr"]
    w       = ext["weights"]
    if M_grid is None:
        M_grid = np.linspace(0.6, 2.6, 200)

    R_band, weight_frac = _mr_band(M_arr, R_arr, w, M_grid, Lam_arr=Lam_arr,
                                   q_low=q_low, q_high=q_high)

    return {
        "label":  label or ext.get("ref", "external"),
        "color":  color,
        "R":      R_band,
        "M_grid": M_grid,
        "weight_frac": weight_frac,
        "N":      len(M_arr),
    }


_REQUIRED_NPY = {
    "cs2":         "cs2.npy",
    "n_long_grid": "n_long_grid.npy",
}
_OPTIONAL_NPY = {
    "mMax":          "mMax.npy",                  # M_TOV per sample, M_sun
    "radius_MASS":   "radius_MASS.npy",           # R(M_grid) per sample, km
    "mass_grid":     "mass_grid.npy",             # M grid in M_sun
    "Delta":         "Delta.npy",                 # pre-computed diagnostics
    "gamma":         "gamma.npy",
    "dc":            "dc.npy",
}


def load_annala_interpolation_ensemble(dir_path,
                                       target_nB_grid=None,
                                       use_published_diagnostics=True,
                                       eps_ref=None, P_ref=None,
                                       eps_ref_native=None, P_ref_native=None,
                                       nB_ref_native=None,
                                       low_density_cs2=None,
                                       verbose=True):
    """
    Load the Annala et al. 2023 interpolation ensemble from a directory
    of .npy files (Zenodo DOI 10.5281/zenodo.10102436).

    The two released datasets (main_text/ and appendix/) follow the
    same layout; pass whichever subdirectory to compare with.
    The Zenodo release is "thinned" -- the data is already a posterior
    subsample under the published combined likelihood (NICER + GW +
    chi-EFT + pQCD at 10 n_s), so uniform weights reproduce the
    published bands exactly.

    Parameters
    ----------
    dir_path : str
        Directory containing cs2.npy, n_long_grid.npy, and optionally
        the M-R, P-eps, and pre-computed-diagnostic files listed in
        _OPTIONAL_NPY.
    target_nB_grid : (L,) array
        Our n_B / n_0 grid.  Pass out['nB_grid'].
    eps_ref_native, P_ref_native : float or None   [PREFERRED]
        Thermodynamic reference point (MeV/fm^3) at nB_ref_native --
        i.e. at the FIRST IN-RANGE TARGET-GRID POINT, the density where
        Annala's supported range begins (~1.03 n_s for the C4 grid).
        When supplied (and the native grid starts above target[0]), the
        Heun integration runs over Annala's supported range ONLY and is
        anchored here, so no assumption about c_s^2 below 1 n_s enters.
        These are the exact npe-mu beta-equilibrium (eps, P) on the same
        chi-EFT curve that defines eps_ref/P_ref, obtained from
        cs2_betaeq_anchors.py via extract_anchor_1n0.py.  This is the
        recommended path and matches how eps_ref/P_ref were produced.
    nB_ref_native : float or None
        Density (in n_0, same units as target_nB_grid) at which
        eps_ref_native/P_ref_native were extracted.  Checked against the
        first in-range target point; if they differ by more than a
        fraction of a grid cell the loader raises, because the grid
        changed and the anchor would otherwise be applied at the wrong
        density (exactly the class of bug this replaced).  Re-run
        extract_anchor_1n0.py with the new grid if that happens.
    eps_ref, P_ref : float
        Thermodynamic reference point (MeV/fm^3) at target_nB_grid[0]
        (0.5 n_s).  Pass out['ref_point']['eps_ref'] / ['P_ref'].
        Used as the FALLBACK anchor when eps_ref_native/P_ref_native are
        not supplied: the integration then starts at target[0] and the
        sub-Annala gap [target[0], native_min) is bridged with a chi-EFT
        c_s^2 curve (see low_density_cs2).  Correct to a few percent, but
        the native-anchor path above is exact.  When the native grid
        already covers target[0] (no out-of-range points) this is simply
        the boundary condition and no bridging occurs.

        Do NOT integrate the in-range subgrid while feeding it the
        target[0] (0.5 n_s) reference: that applies the 0.5-n_s constant
        at ~1 n_s, roughly halving eps and mu_B across the ensemble and
        corrupting mu_B = (eps+P)/n_B and every mu-based diagnostic
        (p/p_free, Delta(mu_B), the recomputed Delta band).  Both paths
        below apply their anchor at the density it belongs to.
    low_density_cs2 : (L,) array or None
        FALLBACK only (ignored when eps_ref_native/P_ref_native are
        given).  Shared low-density (chi-EFT) c_s^2 on target_nB_grid,
        used to bridge the [target[0], native_min) gap.  Below ~1 n_s all
        these ensembles share the same chi-EFT outer core, so a natural
        choice is this work's median c_s^2, e.g.
            low_density_cs2 = np.nanmedian(_to_np(samples_phys), axis=0)
        If None the gap is flat-filled from Annala's own edge value
        (accurate to a few percent).  Has no effect when the native grid
        already covers target[0].
    use_published_diagnostics : bool
        If True (default) and dc.npy / gamma.npy / Delta.npy are present,
        return Annala's diagnostic arrays directly instead of recomputing
        from c_s^2 with our Heun integrator.  The GP loader has no
        published-diagnostic arrays and always recomputes; the d_c
        definition is identical, so the two overlays stay
        comparable either way.
    verbose : bool
        Print summary info.

    Returns
    -------
    ext : dict shaped like the output of load_annala_ensemble:
        cs2_arr, P_arr, eps_arr  : (N, L) on our grid
        weights                  : (N,) uniform (= 1)
        native_grid              : (L_n,) Annala's n grid in n_s
        target_grid              : (L,) our grid
        N                        : int
        ref                      : citation string
        # extras specific to the interpolation ensemble:
        gamma_published, Delta_published, dc_published : (N, L) or None
        M_TOV                    : (N,) if mMax.npy was present, else NaN
        M_arr, R_arr, Lambda_arr : per-sample TOV arrays if available,
                                   else empty object arrays
    """
    if target_nB_grid is None:
        raise ValueError("target_nB_grid is required (pass out['nB_grid']).")
    target = np.asarray(target_nB_grid, dtype=np.float64)

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"directory not found: {dir_path}")

    arrays = {}
    for key, fname in _REQUIRED_NPY.items():
        path_local  = os.path.join(dir_path, fname)
        path_parent = os.path.join(os.path.dirname(os.path.normpath(dir_path)),
                                   fname)
        if os.path.exists(path_local):
            arrays[key] = np.load(path_local)
        elif key == "n_long_grid" and os.path.exists(path_parent):
            arrays[key] = np.load(path_parent)
            if verbose:
                print(f"  found shared n_long_grid.npy in parent: "
                      f"{path_parent}")
        else:
            raise FileNotFoundError(
                f"required file '{fname}' not found in {dir_path} "
                f"or its parent directory")

    for key, fname in _OPTIONAL_NPY.items():
        path = os.path.join(dir_path, fname)
        if os.path.exists(path):
            arrays[key] = np.load(path)

    cs2_native = np.asarray(arrays["cs2"], dtype=np.float64)
    n_native   = np.asarray(arrays["n_long_grid"], dtype=np.float64)

    if cs2_native.ndim != 2:
        raise ValueError(f"cs2.npy must be 2-D (N, L); got {cs2_native.shape}")
    N, L_n = cs2_native.shape
    if n_native.size != L_n:
        raise ValueError(f"n_long_grid length {n_native.size} != cs2 second "
                         f"axis {L_n}")

    if verbose:
        print(f"  load_annala_interpolation_ensemble: {dir_path}")
        print(f"    samples: {N}")
        print(f"    native grid: {L_n} pts in [{n_native.min():.3f}, "
              f"{n_native.max():.3f}] n_s")
        print(f"    target grid: {len(target)} pts in [{target.min():.3f}, "
              f"{target.max():.3f}] n_s")
        opt_present = sorted(k for k in _OPTIONAL_NPY if k in arrays)
        if opt_present:
            print(f"    optional files present: {', '.join(opt_present)}")
        opt_missing = sorted(k for k in _OPTIONAL_NPY if k not in arrays)
        if opt_missing:
            print(f"    optional files NOT found: {', '.join(opt_missing)}")


    in_range = (target >= n_native.min()) & (target <= n_native.max())
    n_oor = int((~in_range).sum())
    if verbose and n_oor > 0:
        print(f"    {n_oor}/{len(target)} target points outside Annala's "
              f"grid -- those slots will be NaN.")

    cs2_arr = np.full((N, len(target)), np.nan)
    for i in range(N):
        cs2_arr[i, in_range] = np.interp(
            target[in_range], n_native, cs2_native[i])

    causal = np.full(N, True)
    for i in range(N):
        c = cs2_arr[i, in_range]
        if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
            causal[i] = False

    n_drop = int((~causal).sum())
    if verbose and n_drop > 0:
        print(f"    dropping {n_drop}/{N} samples that fail causality / "
              f"finite check.")
    cs2_arr = cs2_arr[causal]
    N_keep  = cs2_arr.shape[0]
    L       = len(target)
    idx_in  = np.where(in_range)[0]
    first_in, last_in = idx_in[0], idx_in[-1]


    _ANCHOR_NB_TOL = 0.02
    use_native_anchor = (first_in > 0
                         and eps_ref_native is not None
                         and P_ref_native is not None)

    P_arr   = np.full((N_keep, L), np.nan)
    eps_arr = np.full((N_keep, L), np.nan)

    if use_native_anchor:
        tgt_in = target[in_range]
        if nB_ref_native is not None:
            drift = abs(float(tgt_in[0]) - float(nB_ref_native))
            if drift > _ANCHOR_NB_TOL:
                raise ValueError(
                    f"native anchor was extracted at nB_ref_native="
                    f"{float(nB_ref_native):.4f} n_s but the first in-range "
                    f"target point is {float(tgt_in[0]):.4f} n_s "
                    f"(drift {drift:.4f} > {_ANCHOR_NB_TOL:.4f} n_s).  The "
                    f"target grid changed, so the anchor would be applied at "
                    f"the wrong density -- re-run extract_anchor_1n0.py with "
                    f"this grid and pass the new eps_ref_native / P_ref_native "
                    f"/ nB_ref_native.")
        cs2_in = cs2_arr[:, in_range]
        P_in, eps_in = compute_P_eps_ensemble(
            cs2_in, tgt_in, eps_ref=eps_ref_native, P_ref=P_ref_native,
            verbose=False)
        P_arr[:, in_range]   = P_in
        eps_arr[:, in_range] = eps_in
        if verbose:
            print(f"    anchoring C4 at its first in-range target point "
                  f"{float(tgt_in[0]):.4f} n_s with exact chi-EFT beta-eq "
                  f"reference eps={eps_ref_native:.3f}, P={P_ref_native:.3f} "
                  f"MeV/fm^3 (Option B, integrating in-range support only).")
    else:
        if eps_ref is None or P_ref is None:
            raise ValueError(
                "load_annala_interpolation_ensemble: supply either the native "
                "anchor (eps_ref_native, P_ref_native) or the target[0] "
                "reference (eps_ref, P_ref).  Pass out['ref_point']['eps_ref']"
                " / ['P_ref'] for the latter.")
        span_hi  = last_in + 1                  # integrate up to native_max
        tgt_span = target[:span_hi]
        cs2_native_keep = cs2_native[causal]
        cs2_span = np.empty((N_keep, span_hi), dtype=np.float64)
        for i in range(N_keep):
            cs2_span[i] = np.interp(tgt_span, n_native, cs2_native_keep[i])
        if first_in > 0:
            if low_density_cs2 is not None:
                ld = np.asarray(low_density_cs2, dtype=np.float64)
                if ld.shape[0] != L:
                    raise ValueError(
                        f"low_density_cs2 must be length {L} (target grid); "
                        f"got {ld.shape[0]}.")
                cs2_span[:, :first_in] = ld[None, :first_in]
                gap_src = "shared chi-EFT c_s^2 (low_density_cs2)"
            else:
                cs2_span[:, :first_in] = cs2_span[:, first_in][:, None]
                gap_src = ("flat-fill from Annala's native edge (pass "
                           "eps_ref_native/P_ref_native to anchor exactly)")
            if verbose:
                print(f"    integrating from target[0]={target[0]:.3f} n_s; "
                      f"bridging {first_in} sub-native point(s) below "
                      f"{n_native.min():.3f} n_s with {gap_src} (Option A).")
        P_span, eps_span = compute_P_eps_ensemble(
            cs2_span, tgt_span, eps_ref=eps_ref, P_ref=P_ref, verbose=False)
        P_arr[:, :span_hi]   = P_span
        eps_arr[:, :span_hi] = eps_span


    P_arr[:, ~in_range]   = np.nan
    eps_arr[:, ~in_range] = np.nan


    def _regrid_native(arr_native):
        if arr_native is None:
            return None
        a = np.asarray(arr_native, dtype=np.float64)[causal]
        out = np.full((a.shape[0], len(target)), np.nan)
        for i in range(a.shape[0]):
            out[i, in_range] = np.interp(target[in_range], n_native, a[i])
        return out

    gamma_pub = _regrid_native(arrays.get("gamma"))
    Delta_pub = _regrid_native(arrays.get("Delta"))
    dc_pub    = _regrid_native(arrays.get("dc"))


    if "mMax" in arrays:
        M_TOV = np.asarray(arrays["mMax"], dtype=np.float64)[causal]
    else:
        M_TOV = np.full(cs2_arr.shape[0], np.nan)


    if "radius_MASS" in arrays and "mass_grid" in arrays:
        mass_grid = np.asarray(arrays["mass_grid"], dtype=np.float64)
        R_R = np.asarray(arrays["radius_MASS"], dtype=np.float64)[causal]
        if R_R.shape[1] != mass_grid.size:
            raise ValueError(
                f"radius_MASS columns {R_R.shape[1]} != mass_grid len "
                f"{mass_grid.size}")
        M_obj   = np.array([mass_grid.copy() for _ in range(R_R.shape[0])],
                           dtype=object)
        R_obj   = np.array([R_R[i].copy() for i in range(R_R.shape[0])],
                           dtype=object)
        Lam_obj = np.array([np.full_like(mass_grid, np.nan)
                            for _ in range(R_R.shape[0])], dtype=object)
    else:
        M_obj   = np.array([], dtype=object)
        R_obj   = np.array([], dtype=object)
        Lam_obj = np.array([], dtype=object)

    weights = np.ones(cs2_arr.shape[0], dtype=np.float64)

    if verbose:
        print(f"    kept {cs2_arr.shape[0]} samples; uniform weights "
              f"(thinned posterior sub-sample)")

    return {
        "cs2_arr":           cs2_arr,
        "P_arr":              P_arr,
        "eps_arr":            eps_arr,
        "weights":            weights,
        "native_grid":        n_native,
        "target_grid":        target,
        "M_arr":              M_obj,
        "R_arr":              R_obj,
        "Lambda_arr":         Lam_obj,
        "M_TOV":              M_TOV,
        "N":                  int(cs2_arr.shape[0]),
        "ref": ("Annala et al. 2023 (Nat. Commun. 14, 8451) "
                "-- interpolation ensemble"),
        "in_range":           in_range,
        "gamma_published":    gamma_pub if use_published_diagnostics else None,
        "Delta_published":    Delta_pub if use_published_diagnostics else None,
        "dc_published":       dc_pub    if use_published_diagnostics else None,
    }