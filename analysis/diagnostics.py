"""
Derived-quantity computations from posterior c_s^2 ensembles.

Pure post-processing on the dict produced by eos_sampling.run_sampling
(loaded from a .pt or held in memory).  No new sampling, no new TOV,
no pQCD recomputation.  The thermodynamic integrator and TOV stable-
branch logic come from eos_sampling.reweighting -- this module only
adds derived quantities and their weighted posterior summaries.

"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import brentq

# Reuse the validated Heun integrator and TOV-postprocessing utilities
# from the existing pipeline.
from eos_sampling.reweighting import (
    thermodynamic_integration_np,
    weighted_quantile,
    _stable_branch,
    N0_DEFAULT,
)

def _to_np(x):
    """Tolerant tensor->numpy conversion (mirrors plotting._to_np)."""
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)


def _quantile_band(arr2d, w, q_low=0.16, q_high=0.84):
    """Pointwise (q_low, median, q_high) down axis 0; NaN where <2 finite, positive-weight points."""
    arr2d = np.asarray(arr2d, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    L = arr2d.shape[1]
    band = np.full((3, L), np.nan)
    for j in range(L):
        col = arr2d[:, j]
        mask = np.isfinite(col) & (w > 0)
        if mask.sum() < 2:
            continue
        band[0, j] = weighted_quantile(col[mask], w[mask], q_low)
        band[1, j] = weighted_quantile(col[mask], w[mask], 0.50)
        band[2, j] = weighted_quantile(col[mask], w[mask], q_high)
    return band


def _mr_band(M_arr, R_arr, weights, M_grid, *, Lam_arr=None,
             q_low=0.16, q_high=0.84):
    """Stable-branch R(M) on M_grid -> ((3, L) quantile band, per-M support weight fraction)."""
    M = _to_np(M_arr)
    R = _to_np(R_arr)
    w = np.asarray(weights, dtype=np.float64)
    N = len(M)
    L = len(M_grid)
    R_on_grid = np.full((N, L), np.nan)
    has_support = np.zeros((N, L), dtype=bool)
    for i in range(N):
        M_i = np.asarray(M[i], dtype=np.float64)
        R_i = np.asarray(R[i], dtype=np.float64)
        L_i = (np.zeros_like(M_i) if Lam_arr is None
               else np.asarray(Lam_arr[i], dtype=np.float64))
        M_s, R_s, _ = _stable_branch(M_i, R_i, L_i)
        if M_s.size < 2:
            continue
        valid = (M_grid >= M_s.min()) & (M_grid <= M_s.max())
        if not valid.any():
            continue
        R_on_grid[i, valid] = np.interp(M_grid[valid], M_s, R_s)
        has_support[i, valid] = True
    wn = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / w.size)
    weight_frac = (wn[:, None] * has_support).sum(axis=0)
    return _quantile_band(R_on_grid, w, q_low, q_high), weight_frac


def compute_P_eps_ensemble(samples_phys, nB_grid, n0=N0_DEFAULT,
                           eps_ref=None, P_ref=None, verbose=True):
    """
    Compute (P, eps) on the n_B grid for every sample in the ensemble.

    Wrapper around the validated Heun integrator
    eos_sampling.reweighting.thermodynamic_integration_np.

    Parameters
    ----------
    samples_phys : (N, L) torch.Tensor or numpy array of c_s^2 samples
                   in physical units.
    nB_grid      : (L,) grid in n_B/n_0 units.
    n0           : saturation density in fm^-3 (default 0.16).
    eps_ref, P_ref : reference point (MeV/fm^3) at the grid lower edge.
                     REQUIRED -- read them from the .pt produced by the
                     pipeline (ref = out['ref_point']).  No default.

    Returns
    -------
    P_arr   : (N, L) numpy array of pressure       in MeV/fm^3.
    eps_arr : (N, L) numpy array of energy density in MeV/fm^3.
    """
    cs2 = _to_np(samples_phys).astype(np.float64)
    nB  = _to_np(nB_grid).astype(np.float64)

    nB_phys = nB * n0  # fm^-3

    if eps_ref is None or P_ref is None:
        raise ValueError(
            "compute_P_eps_ensemble: eps_ref and P_ref must be passed "
            "explicitly (MeV/fm^3 at nB_grid[0]).  Read them from the "
            ".pt: ref = out['ref_point'], then pass "
            "eps_ref=ref['eps_ref'], P_ref=ref['P_ref'].  No default.")

    N, L = cs2.shape
    P_arr   = np.empty((N, L), dtype=np.float64)
    eps_arr = np.empty((N, L), dtype=np.float64)

    for n in range(N):
        P_arr[n], eps_arr[n] = thermodynamic_integration_np(
            cs2[n], nB_phys, eps_ref, P_ref)

    if verbose:
        print(f"  compute_P_eps_ensemble: integrated {N} samples on "
              f"{L}-point n_B grid")
        print(f"    P   range:  [{P_arr.min():.2f}, {P_arr.max():.2f}] "
              f"MeV/fm^3")
        print(f"    eps range:  [{eps_arr.min():.2f}, {eps_arr.max():.2f}] "
              f"MeV/fm^3")

    return P_arr, eps_arr


def compute_diagnostics(P_arr, eps_arr, cs2_arr):
    """
    Compute (gamma, Delta, d_c) on the n_B grid for every sample.

    Definitions
    -----------
    gamma : polytropic index, gamma = d ln P / d ln eps = c_s^2 * eps / P.
        Annala et al. (Nature Physics 2020) proposed this as a quark-matter
        diagnostic: gamma -> 1 in the conformal limit, gamma in [1, 1.7]
        for quark matter, gamma >= 2.5 for hadronic matter.  An alternative
        definition gamma_n = d ln P / d ln n_B coincides with the one used
        here only at the conformal limit; we use the eps version because
        that's the deconfinement-diagnostic version.

    Delta : trace anomaly, Delta = 1/3 - P/eps.
        Fujimoto, Fukushima, McLerran, Praszalowicz (Phys. Rev. Lett. 129,
        252702, 2022).  Dimensionless conformality measure;
        Delta -> 0 in the conformal limit.

    d_c   : composite conformal-limit measure of Annala et al.
        (Nature Communications 14, 8451, 2023):
            d_c = sqrt(Delta^2 + (Delta')^2),
            Delta' = d Delta / d ln eps.
        Closed-form for Delta':  Delta' = P/eps - c_s^2.
        Derivation:
            d Delta / d ln eps = eps * d(1/3 - P/eps)/d eps
                               = eps * (P/eps^2 - c_s^2/eps)
                               = P/eps - c_s^2.

    Parameters
    ----------
    P_arr, eps_arr, cs2_arr : (N, L) arrays
        Pressure (MeV/fm^3), energy density (MeV/fm^3), c_s^2.
        Tensor or numpy-array inputs both accepted.

    Returns
    -------
    gamma_arr, Delta_arr, dc_arr : each (N, L) numpy arrays.
    """
    P   = _to_np(P_arr).astype(np.float64)
    eps = _to_np(eps_arr).astype(np.float64)
    cs2 = _to_np(cs2_arr).astype(np.float64)

    # Guards against division by zero at the very low-density edge.
    # The grid starts at n_B / n_0 = 0.5 where P ~ 0.42 MeV/fm^3 (chEFT),
    P_safe   = np.where(P   > 1e-12, P,   1e-12)
    eps_safe = np.where(eps > 1e-12, eps, 1e-12)

    gamma_arr  = cs2 * eps_safe / P_safe
    P_over_eps = P / eps_safe
    Delta_arr  = (1.0 / 3.0) - P_over_eps
    Delta_p    = P_over_eps - cs2          # closed-form derivative
    dc_arr     = np.sqrt(Delta_arr**2 + Delta_p**2)

    return gamma_arr, Delta_arr, dc_arr

def fiducial_NS_quantities(M_arr, R_arr, L_arr, weights,
                           target_masses=(1.4, 2.0)):
    """
    Weighted posteriors of R(M_t), Lambda(M_t), and M_TOV from the cached
    TOV arrays out['reweighting']['M' / 'R' / 'Lambda'].

    Per-sample interpolation along the strictly-monotonic stable branch
    (eos_sampling.reweighting._stable_branch) gives R(M_t) and Lambda(M_t);
    samples whose stable branch does not span M_t contribute NaN and are
    automatically dropped from the weighted quantile.

    Parameters
    ----------
    M_arr, R_arr, L_arr : (N, n_central) numpy arrays.
    weights             : (N,) importance weights (need not be normalised).
    target_masses       : iterable of M/M_sun values to evaluate at.

    Returns
    -------
    fids : dict.  For each M_t in target_masses,
        fids[M_t] = {
            "R_q16", "R_q50", "R_q84"   in km,
            "L_q16", "L_q50", "L_q84"   dimensionless,
            "n_valid"                   number of samples whose stable
                                        branch reached M_t.
        }
        Plus a top-level entry
        fids["M_TOV"] = {"q16", "q50", "q84", "n_valid"} in M_sun.
    """
    M = _to_np(M_arr).astype(np.float64)
    R = _to_np(R_arr).astype(np.float64)
    L = _to_np(L_arr).astype(np.float64)
    w = _to_np(weights).astype(np.float64)
    N = M.shape[0]

    fids = {}
    for M_t in target_masses:
        R_at = np.full(N, np.nan)
        L_at = np.full(N, np.nan)
        for n in range(N):
            M_s, R_s, L_s = _stable_branch(M[n], R[n], L[n])
            if M_s.size < 2:
                continue
            if (M_t < M_s.min()) or (M_t > M_s.max()):
                continue
            R_at[n] = float(np.interp(M_t, M_s, R_s))
            L_at[n] = float(np.interp(M_t, M_s, L_s))
        fids[M_t] = {
            "R_q16":   weighted_quantile(R_at, w, 0.16),
            "R_q50":   weighted_quantile(R_at, w, 0.50),
            "R_q84":   weighted_quantile(R_at, w, 0.84),
            "L_q16":   weighted_quantile(L_at, w, 0.16),
            "L_q50":   weighted_quantile(L_at, w, 0.50),
            "L_q84":   weighted_quantile(L_at, w, 0.84),
            "n_valid": int(np.isfinite(R_at).sum()),
            "w_valid": (float(w[np.isfinite(R_at)].sum() / w.sum())
                        if w.sum() > 0 else float("nan")),
        }

    M_TOV_arr = np.array([
        np.nanmax(M[n]) if np.any(np.isfinite(M[n])) else np.nan
        for n in range(N)])
    fids["M_TOV"] = {
        "q16":     weighted_quantile(M_TOV_arr, w, 0.16),
        "q50":     weighted_quantile(M_TOV_arr, w, 0.50),
        "q84":     weighted_quantile(M_TOV_arr, w, 0.84),
        "n_valid": int(np.isfinite(M_TOV_arr).sum()),
        "w_valid": (float(w[np.isfinite(M_TOV_arr)].sum() / w.sum())
                    if w.sum() > 0 else float("nan")),
    }
    return fids


def print_fiducial_table(fids):
    """Pretty-print the dict returned by fiducial_NS_quantities."""
    print(f"  {'Target':>12s}  {'q16':>10s}  {'q50':>10s}  {'q84':>10s}  "
          f"{'N_valid':>8s}  {'W_valid':>8s}")
    print(f"  {'-' * 70}")
    for M_t in [k for k in fids if isinstance(k, (int, float))]:
        f = fids[M_t]
        print(f"  R({M_t:g} M_sun) [km] : "
              f"{f['R_q16']:8.3f}  {f['R_q50']:8.3f}  {f['R_q84']:8.3f}  "
              f"{f['n_valid']:8d}  {f['w_valid']:8.3f}")
        print(f"  L({M_t:g} M_sun)      : "
              f"{f['L_q16']:8.1f}  {f['L_q50']:8.1f}  {f['L_q84']:8.1f}  "
              f"{f['n_valid']:8d}  {f['w_valid']:8.3f}")
    if "M_TOV" in fids:
        f = fids["M_TOV"]
        print(f"  M_TOV [M_sun]      : "
              f"{f['q16']:8.3f}  {f['q50']:8.3f}  {f['q84']:8.3f}  "
              f"{f['n_valid']:8d}  {f['w_valid']:8.3f}")

def phase_transition_diagnostic(samples_phys, weights, nB_grid,
                                threshold=0.05, window=(1.5, 6.0)):
    """
    Posterior probability that c_s^2 dips below `threshold` somewhere in
    the n_B/n_0 window *and* the dip lies at a higher density than the
    in-window sound-speed peak (n(c_s^2,min) > n(c_s^2,max)) -- a model-
    independent flag for first-order-like softening (Maxwell coexistence
    in Class 5 of the training prior, but the diagnostic is shape-based,
    not class-based).

    The two conditions together are the necessary condition of Brandes
    et al.: a low c_s^2 (the transition fingerprint) that follows the
    peak (excluding curves merely soft at low density).  It remains a
    *necessary, not sufficient* flag -- it does not measure the
    coexistence width Delta n/n and so does NOT by itself certify a
    *strong* first-order transition.

    Also returns the (n_peak, c_s2_peak) joint distribution -- a 2D
    posterior summary that pinpoints where, and how high, the speed-of-
    sound peak sits.

    Parameters
    ----------
    samples_phys : (N, L) c_s^2 ensemble.
    weights      : (N,) importance weights.
    nB_grid      : (L,) grid in n_B/n_0.
    threshold    : c_s^2 floor below which a sample is flagged as
                   exhibiting phase-transition-like softening.  0.05 is
                   well below the lowest values reached by smooth
                   crossover (Class 7) curves but well above numerical
                   floor.
    window       : (n_low, n_high) in n_B/n_0 in which to apply the
                   threshold.  Default (1.5, 6.0) covers the canonical
                   neutron-star core regime.

    Returns
    -------
    diag : dict
        P_PT       : weighted fraction of samples that both dip below
                     `threshold` inside the window and have the dip at a
                     higher density than the in-window peak  (-> Bayes-
                     factor numerator for a "phase-transition feature is
                     present" hypothesis when divided by the prior
                     fraction; see analysis notebook for the full
                     prior/posterior comparison).
        has_dip    : (N,) bool, the per-sample PT flag (dip + ordering).
        n_dip      : (N,) density n_B/n_0 of the in-window c_s^2 minimum.
        peak_n     : (N,) per-sample peak location (n_B/n_0).
        peak_cs2   : (N,) per-sample peak c_s^2.
        weights    : (N,) renormalised weights for direct plotting.
        threshold, window : echoed for traceability.
    """
    cs2 = _to_np(samples_phys).astype(np.float64)
    nB  = _to_np(nB_grid).astype(np.float64)
    w   = _to_np(weights).astype(np.float64)
    if w.sum() > 0:
        w_n = w / w.sum()
    else:
        w_n = np.full_like(w, 1.0 / w.size)

    N, L = cs2.shape

    in_win = (nB >= window[0]) & (nB <= window[1])
    if in_win.sum() < 2:
        raise ValueError(f"window {window} contains fewer than 2 grid points")

    cs2_win  = cs2[:, in_win]
    win_pos  = np.where(in_win)[0]                  # window points, full-grid indices
    dip_pos  = win_pos[np.argmin(cs2_win, axis=1)]  # location of the in-window minimum
    pk_pos   = win_pos[np.argmax(cs2_win, axis=1)]  # location of the in-window peak
    # Brandes et al. necessary condition (both prongs): c_s^2 dips below
    # `threshold` inside the window AND the dip follows the (in-window) sound-
    # speed peak, i.e. n(c_s^2,min) > n(c_s^2,max).  The ordering removes curves
    # that are merely soft at low density and never rise-then-drop.
    has_dip  = (cs2[np.arange(N), dip_pos] < threshold) & (nB[dip_pos] > nB[pk_pos])
    P_PT     = float(np.sum(w_n[has_dip]))
    n_dip    = nB[dip_pos]

    peak_idx = np.argmax(cs2, axis=1)
    peak_n   = nB[peak_idx]
    peak_cs2 = cs2[np.arange(N), peak_idx]

    return {
        "P_PT":      P_PT,
        "has_dip":   has_dip,
        "n_dip":     n_dip,
        "peak_n":    peak_n,
        "peak_cs2":  peak_cs2,
        "weights":   w_n,
        "threshold": threshold,
        "window":    window,
    }

_HBARC      = 197.3269804              # MeV fm
_MN, _MP    = 939.565, 938.272         # MeV
_ME, _MMU   = 0.51099895, 105.6583755  # MeV


def _lep_n(mu, m):
    """Lepton number density [fm^-3] for chemical potential mu, mass m."""
    return 0.0 if mu <= m else (mu*mu - m*m)**1.5 / (3*np.pi**2*_HBARC**3)


def _lep_eps(mu, m):
    """Lepton energy density [MeV/fm^3]: relativistic Fermi gas (g=2)."""
    if mu <= m:
        return 0.0
    pF = np.sqrt(mu*mu - m*m)
    return (pF*(m*m + 2*pF*pF)*mu - m**4*np.arcsinh(pF/m)) / (8*np.pi**2*_HBARC**3)



def symmetric_matter_baseline(nB_grid, n0=N0_DEFAULT, buqeye_dir="chEFT",
                              Lambda=500):
    """
    Symmetric nuclear matter E/A [MeV] interpolated onto nB_grid (in units
    of n/n0).  Loads the BUQEYE posterior-mean E0(nB) from `buqeye_dir`.
    """
    f_E0 = os.path.join(buqeye_dir, f"EA_SNM_Lambda-{Lambda}_samples_N3LO.npy")
    f_nB = os.path.join(buqeye_dir, f"nB_density_Lambda-{Lambda}.npy")
    if not (os.path.exists(f_E0) and os.path.exists(f_nB)):
        raise FileNotFoundError(
            f"[L extraction] BUQEYE symmetric-matter baseline not found.\n"
            f"  Expected:\n    {f_E0}\n    {f_nB}\n"
            f"  Generate them with chEFT/cs2_betaeq_anchors.py, or pass "
            f"buqeye_dir=.../Lambda=... pointing to where they live.")
    E0_mean = np.nanmean(np.load(f_E0), axis=0)        # (M,) MeV, no rest mass
    nB_buq  = np.load(f_nB)                            # (M,) fm^-3
    print(f"  [L extraction] E0 baseline = BUQEYE chi-EFT "
          f"(Lambda={Lambda}, {os.path.basename(f_E0)})")
    return np.interp(np.asarray(nB_grid, float) * n0, nB_buq, E0_mean)


def _solve_Esym_xp(nB, eps_recon, E0_val):
    """
    Solve (E_sym, x_p) at one (density, sample) from the reconstructed total
    beta-eq energy density together with beta-equilibrium + charge neutrality.
    """
    def xp_of(s):
        def r1(x):
            mu_e = (_MN - _MP) + 4 * s * (1 - 2 * x)
            return x * nB - (_lep_n(mu_e, _ME) + _lep_n(mu_e, _MMU))
        return brentq(r1, 1e-9, 0.49)

    def res(s):
        x = xp_of(s)
        mu_e = (_MN - _MP) + 4 * s * (1 - 2 * x)
        eps_model = (nB * ((1 - x) * _MN + x * _MP + E0_val + s * (1 - 2 * x)**2)
                     + _lep_eps(mu_e, _ME) + _lep_eps(mu_e, _MMU))
        return eps_model - eps_recon

    try:
        s = brentq(res, 1.0, 250.0)
        return s, xp_of(s)
    except Exception:
        return np.nan, np.nan


def compute_L_per_sample_textmethod(eps_arr, nB_grid, E0_grid, *,
                                    n0=N0_DEFAULT, fit_lo=0.75, fit_hi=1.25,
                                    return_JLK=False):
    """
    Per-sample symmetry-energy slope L [MeV] via the text procedure
    (Esym_extract + parabolic fit).

    Parameters
    ----------
    eps_arr : (N, Lgrid) total beta-eq energy density
              [MeV/fm^3, INCLUDING rest mass -- ~150 near n0].
    nB_grid : (Lgrid,) grid in units of n/n0.
    E0_grid : (Lgrid,) SNM E/A [MeV], e.g. from symmetric_matter_baseline.

    For each sample, E_sym(n) is recovered by inverting the total beta-eq
    energy density (subtracting the lepton contribution), then fit to
    E_sym = J + L*chi + 1/2 Ksym*chi^2 with chi = (n - n0)/(3 n0) over the
    [fit_lo, fit_hi]*n0 window; L is the chi-coefficient (= 3 n0 dE_sym/dn|n0).

    Inverse-variance (diagonal) weighting from the ensemble keeps the fit
    stable where the diffusion density-density covariance is rank-deficient.
    Samples with any non-finite E_sym in the window return NaN.
    With return_JLK=True, also returns the intercept J = E_sym(n0) and the
    curvature Ksym (both already computed by the same fit) as (L, J, Ksym).
    """
    eps_arr = np.asarray(eps_arr, float)
    nB_phys = np.asarray(nB_grid, float) * n0
    win = (np.asarray(nB_grid) >= fit_lo) & (np.asarray(nB_grid) <= fit_hi)
    chi = (nB_phys[win] - n0) / (3 * n0)
    A = np.vstack([np.ones_like(chi), chi, 0.5 * chi**2]).T  # J + L*chi + 1/2 Ksym*chi^2
    N = eps_arr.shape[0]
    idx = np.where(win)[0]
    Es = np.full((N, idx.size), np.nan)
    for nn in range(N):
        for j, k in enumerate(idx):
            Es[nn, j], _ = _solve_Esym_xp(nB_phys[k], eps_arr[nn, k], E0_grid[k])
    # inverse-variance (diagonal) weight from the diffusion ensemble -- stable
    var = np.nanvar(Es, axis=0)
    W = np.diag(1.0 / np.clip(var, 1e-9, None))
    AtW = A.T @ W
    M = np.linalg.inv(AtW @ A) @ AtW            # weighted-LS estimator matrix
    ok = np.all(np.isfinite(Es), axis=1)
    coef = np.full((N, 3), np.nan)              # columns: J, L, Ksym
    coef[ok] = Es[ok] @ M.T
    L_per = coef[:, 1]                          # coefficient of chi == L
    if return_JLK:
        return L_per, coef[:, 0], coef[:, 2]    # L, J, Ksym  [MeV]
    return L_per

def symmetric_matter_baseline_draws(nB_grid, n0=N0_DEFAULT, buqeye_dir="chEFT",
                                    Lambda=500):
    """
    Ensemble of BUQEYE symmetric-nuclear-matter E/A [MeV] curves interpolated
    onto nB_grid (in units of n/n0).

    Same inputs and files as `symmetric_matter_baseline`, but returns EVERY
    posterior draw rather than collapsing them with np.nanmean, so the chi-EFT
    truncation uncertainty of symmetric matter can be marginalized over in the
    L extraction.

    Returns
    -------
    E0_draws : (n_draws, len(nB_grid)) array, no rest mass.
               Draws with fewer than two finite baseline points are dropped.
    """
    f_E0 = os.path.join(buqeye_dir, f"EA_SNM_Lambda-{Lambda}_samples_N3LO.npy")
    f_nB = os.path.join(buqeye_dir, f"nB_density_Lambda-{Lambda}.npy")
    if not (os.path.exists(f_E0) and os.path.exists(f_nB)):
        raise FileNotFoundError(
            f"[L extraction] BUQEYE symmetric-matter baseline not found.\n"
            f"  Expected:\n    {f_E0}\n    {f_nB}\n"
            f"  Generate them with chEFT/cs2_betaeq_anchors.py, or pass "
            f"buqeye_dir=.../Lambda=... pointing to where they live.")

    E0_samples = np.load(f_E0)                 # (n_draws, M) MeV, no rest mass
    nB_buq     = np.load(f_nB)                 # (M,) fm^-3
    if E0_samples.ndim == 1:                   # a single stored curve
        E0_samples = E0_samples[None, :]

    target = np.asarray(nB_grid, float) * n0   # fm^-3
    rows = []
    for row in E0_samples:
        m = np.isfinite(row) & np.isfinite(nB_buq)
        if m.sum() < 2:
            continue
        # np.interp needs increasing xp; masking only drops points, preserving order
        rows.append(np.interp(target, nB_buq[m], row[m]))
    if not rows:
        raise ValueError("[L extraction] no usable SNM baseline draws "
                         "(every row had < 2 finite points).")

    E0_draws = np.vstack(rows)
    print(f"  [L extraction] E0 baseline ENSEMBLE = BUQEYE chi-EFT "
          f"(Lambda={Lambda}, {E0_draws.shape[0]} draws, "
          f"{os.path.basename(f_E0)})")
    return E0_draws


def compute_L_per_sample_marginalized(eps_arr, nB_grid, E0_draws, *,
                                      n0=N0_DEFAULT, fit_lo=0.75, fit_hi=1.25,
                                      seed=0, return_pairing=False):
    """
    Per-sample symmetry-energy slope L [MeV], marginalized over the chi-EFT
    truncation uncertainty of the symmetric-matter baseline.

    The inversion and parabolic fit are IDENTICAL to
    `compute_L_per_sample_textmethod`.  The only difference: instead of one
    shared baseline curve, each EOS sample is paired with a single SNM
    baseline curve drawn (with replacement, reproducible via `seed`) from
    `E0_draws`.  Pooling the resulting per-sample L values therefore samples
    the joint posterior

        P(L) = int P(L | eos, E0) P(eos) P(E0) d(eos) d(E0),

    i.e. the SNM baseline uncertainty enters the L posterior.  Importance
    weights (if any) are applied afterwards in the weighted quantile exactly
    as for the fixed-baseline result -- the two marginalizations compose.

    Parameters
    ----------
    eps_arr  : (N, Lgrid) total beta-eq energy density
               [MeV/fm^3, INCLUDING rest mass -- ~150 near n0].
    nB_grid  : (Lgrid,) grid in units of n/n0.
    E0_draws : (n_draws, Lgrid) SNM baseline ensemble from
               symmetric_matter_baseline_draws.  A 1-D array is accepted and
               treated as a single draw, in which case this reproduces the
               fixed-baseline compute_L_per_sample_textmethod result.
    seed     : RNG seed for the EOS-sample <-> baseline-draw pairing.
    return_pairing : if True, also return the (N,) int array of draw indices
               used, so a run can be reproduced or audited.

    Returns
    -------
    L_per : (N,) array; NaN where any in-window E_sym is non-finite.
    idx   : (N,) int array of baseline-draw indices  (only if return_pairing).
    """
    eps_arr  = np.asarray(eps_arr, float)
    E0_draws = np.asarray(E0_draws, float)
    if E0_draws.ndim == 1:
        E0_draws = E0_draws[None, :]

    nB_phys = np.asarray(nB_grid, float) * n0
    win = (np.asarray(nB_grid) >= fit_lo) & (np.asarray(nB_grid) <= fit_hi)
    chi = (nB_phys[win] - n0) / (3 * n0)
    A = np.vstack([np.ones_like(chi), chi, 0.5 * chi**2]).T   # J + L*chi + 1/2 Ksym*chi^2

    N        = eps_arr.shape[0]
    idx_grid = np.where(win)[0]
    n_draws  = E0_draws.shape[0]

    rng  = np.random.default_rng(seed)
    pair = rng.integers(0, n_draws, size=N)     # one independent baseline draw per sample

    Es = np.full((N, idx_grid.size), np.nan)
    for nn in range(N):
        E0_row = E0_draws[pair[nn]]
        for j, k in enumerate(idx_grid):
            Es[nn, j], _ = _solve_Esym_xp(nB_phys[k], eps_arr[nn, k], E0_row[k])

    # Same diagonal inverse-variance GLS stabiliser as the fixed-baseline
    # routine.  For any fixed W, the chi-coefficient is an unbiased estimate
    # of L, so W stabilises the fit without biasing L (verified in the test).
    var = np.nanvar(Es, axis=0)
    W   = np.diag(1.0 / np.clip(var, 1e-9, None))
    AtW = A.T @ W
    M   = np.linalg.inv(AtW @ A) @ AtW           # weighted-LS estimator matrix

    L_per = np.full(N, np.nan)
    ok = np.all(np.isfinite(Es), axis=1)
    L_per[ok] = (Es[ok] @ M.T)[:, 1]             # coefficient of chi == L

    if return_pairing:
        return L_per, pair
    return L_per

def compute_L_from_S2_samples(nB_grid=None, *, n0=N0_DEFAULT, buqeye_dir="chEFT",
                              Lambda=500, fit_lo=0.75, fit_hi=1.25,
                              return_JLK=True):
    """
    Covariance-consistent chi-EFT symmetry-energy slope L [MeV], obtained by
    differentiating the BUQEYE beta-equilibrium symmetry-energy samples
    S2_BETAEQ directly -- NOT by inverting the EOS and NOT by pairing E0 draws.

    Each BUQEYE draw carries S2(n) = E_PNM/A - E_SNM/A computed consistently, so
    the neutron- and symmetric-matter chi-EFT truncation errors are correlated
    within a draw and their partial cancellation is automatic.  The spread over
    draws is therefore the covariance-consistent symmetry-energy uncertainty --
    the honest number that the independent-E0-draw route
    (compute_L_per_sample_marginalized) over-counts.

    Loads S2_BETAEQ_Lambda-{Lambda}_samples_N3LO.npy  (shape (n_draws, M), MeV)
    and nB_density_Lambda-{Lambda}.npy               (shape (M,), fm^-3)
    from buqeye_dir.  For each draw, S2(n) is interpolated onto the
    [fit_lo, fit_hi]*n0 window and fit (ordinary least squares -- the S2 curves
    are smooth) to  J + L*chi + 1/2 Ksym*chi^2,  chi = (n - n0)/(3 n0);
    L is the chi-coefficient, identical in definition to
    compute_L_per_sample_textmethod so the two are directly comparable.

    Parameters
    ----------
    nB_grid : optional (Lgrid,) grid in units of n/n0 to fit on.  If None, the
              native BUQEYE density grid points inside the window are used.

    Returns
    -------
    (L, J, Ksym) per-draw arrays [MeV] if return_JLK else L.
    Assumes S2_BETAEQ holds the symmetry energy S(n) on the nB_density grid,
    co-indexed by chiral draw -- verify against the generation notebook.
    """
    f_S2 = os.path.join(buqeye_dir, f"S2_BETAEQ_Lambda-{Lambda}_samples_N3LO.npy")
    f_nB = os.path.join(buqeye_dir, f"nB_density_Lambda-{Lambda}.npy")
    if not (os.path.exists(f_S2) and os.path.exists(f_nB)):
        raise FileNotFoundError(
            f"[L cov-consistent] BUQEYE symmetry-energy samples not found.\n"
            f"  Expected:\n    {f_S2}\n    {f_nB}\n"
            f"  (S2_BETAEQ = symmetry energy S(n) per chiral draw.)")

    S2     = np.load(f_S2)                      # (n_draws, M) MeV
    nB_buq = np.load(f_nB)                      # (M,) fm^-3
    if S2.ndim == 1:
        S2 = S2[None, :]

    target = nB_buq if nB_grid is None else np.asarray(nB_grid, float) * n0
    win = (target >= fit_lo * n0) & (target <= fit_hi * n0)
    if win.sum() < 3:
        raise ValueError(f"[L cov-consistent] only {int(win.sum())} grid points "
                         f"in [{fit_lo}, {fit_hi}] n0; need >= 3 for the parabola.")
    chi = (target[win] - n0) / (3 * n0)
    A = np.vstack([np.ones_like(chi), chi, 0.5 * chi**2]).T   # J + L*chi + 1/2 Ksym*chi^2
    Minv = np.linalg.inv(A.T @ A) @ A.T                       # OLS estimator

    n_draws = S2.shape[0]
    coef = np.full((n_draws, 3), np.nan)                     # columns: J, L, Ksym
    for d in range(n_draws):
        row = S2[d]
        mfin = np.isfinite(row) & np.isfinite(nB_buq)
        if mfin.sum() < 2:
            continue
        Sw = np.interp(target[win], nB_buq[mfin], row[mfin])
        coef[d] = Minv @ Sw

    L, J, Ksym = coef[:, 1], coef[:, 0], coef[:, 2]
    _med = np.nanmedian(L)
    print(f"  [L cov-consistent] from S2_BETAEQ (Lambda={Lambda}, {n_draws} draws): "
          f"L = {_med:.1f} +{np.nanpercentile(L, 84) - _med:.1f}"
          f"/-{_med - np.nanpercentile(L, 16):.1f} MeV  (chi-EFT, correlation intact)")
    if return_JLK:
        return L, J, Ksym
    return L