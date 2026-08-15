"""
pQCD likelihood for the EOS reweighter.

Implements the Komoltsev+2024 marginalized pQCD likelihood
(Zenodo 10.5281/zenodo.15407795): a real-valued log-likelihood that
GRADES samples within the allowed window of high-density extensions,
rather than just flagging them in or out.  Requires the Zenodo HDF5
data file on disk and `eos_marginalization.py` in the same directory;
loaded lazily on first call and cached per process.

The pQCD anchor (mu_H = 2.6 GeV, n_H ~ 35-40 n_sat at the central
scale) is baked into the Zenodo data product.

Dependencies
------------
This module imports three symbols from the base reweighting module:
  thermodynamic_integration_np, PRESS_TO_INV_KM2, N0_DEFAULT
The import is placed at module top-level; reweighting.py imports
pqcd.py lazily (inside exact_log_likelihood) to avoid a circular
import at package load.

Physics
-------
At T = 0, any EOS must satisfy three universal constraints:
  (i)   causality:     c_s^2 = dP/deps = n/(mu dn/dmu) <= 1
  (ii)  stability:     dn/dmu >= 0 (including delta-function jumps
                       at first-order phase transitions)
  (iii) consistency:   dP = n dmu (T = 0 Gibbs-Duhem)

These constraints define a finite window of high-density extensions
that connect the neutron-star termination point (n_T, p_T, mu_T) to
the pQCD anchor.  The Komoltsev+2024 marginalized likelihood scores
each EOS sample by how compatible its termination point is with the
ensemble of allowed extensions.
"""

from __future__ import annotations

import numpy as np

from .reweighting import (thermodynamic_integration_np,
                          central_pressure_grid,
                          N0_DEFAULT)


# ----------------------------------------------------------------
# Module-level cache for the marginalized-likelihood callable
# ----------------------------------------------------------------
# The Zenodo HDF5 data file (~42 MB) is loaded ONCE per Python process
# and reused for all subsequent calls.  When importance_weights runs
# with joblib n_jobs=-1, each worker process maintains its own cache
# copy; the data file is loaded as many times as there are processes
# (typically 4-16 on a workstation), not once per sample.  Process-
# level caching is the right granularity here -- joblib's loky backend
# ships args but not module state across processes, so a thread-level
# cache would be wasted.
#
# Keyed on the resolved absolute path of the data file, so two callers
# that name the same file by different relative paths still share the
# loaded marginalizer within a process.

_MARGINALIZER_CACHE = {}


def _get_marginalizer(data_path):
    """
    Lazy-load + cache the Komoltsev+2024 marginalized-likelihood callable.

    Returns a closure with signature (n0, e0, p0) -> float likelihood,
    matching the `interp_kernels` callable that the Zenodo
    `eos_marginalization.marg_QCD_likelihood()` method returns.  Note
    that this is plain likelihood, NOT log-likelihood -- the log
    transform happens in _call_marginalizer.

    Implementation notes
    --------------------
    The Zenodo class `eos_marginalization` (DOI 10.5281/zenodo.15407795)
    has a quirk: __init__ opens the data file by HARDCODED basename
    relative to the current working directory:

        filename = 'eos_extensions_..._pQCD-25-40.h5'
        self.eos_extensions = pandas.read_hdf(filename)

    To make this work with an arbitrary `data_path`, we (a) verify the
    basename matches, (b) chdir to the data file's parent directory
    around the construction call, and (c) chdir back in a `finally`.
    The `eos_marginalization.py` module file must also live in (or be
    importable from) that directory; we add it to sys.path.

    Construction is expensive: __init__ reads the ~42 MB HDF5 table of
    EOS extensions, and marg_QCD_likelihood() then builds ~840 SciPy
    KDE objects (one per nL slice in [1, 35]*n_sat).  Total wall time
    is typically tens of seconds per joblib worker (~40 s observed).
    The closure is cached in _MARGINALIZER_CACHE keyed on resolved
    absolute path, so this cost is paid once per process per data file.

    Caveats
    -------
    * Only the conditioned likelihood (`flag='conditioned'`) is
      exposed.  The Zenodo class also supports `flag='prior'` (loads
      a separate data file built without the pQCD sound-speed
      conditioning); that is a distinct, more conservative likelihood
      -- not a denominator -- and is only useful as a sensitivity
      ablation.  To enable it we would extend pqcd_config with a
      "flag" key and pass it through to the constructor.

    * One data path per Python process.  Once `eos_marginalization`
      is imported from one directory, Python caches it in sys.modules,
      so passing a different data_path later in the same process will
      not re-import.  The class itself doesn't capture cwd (only its
      __init__ does), so as long as the cache key (abs path) matches
      the actually-loaded data file, this works correctly.

    * Each joblib worker process loads its own copy of the data file
      and KDE table (~80 MB resident per worker).  On an 8-core
      workstation that's ~640 MB total -- usually fine; on a small
      VM consider n_jobs=1 or 2.
    """
    import os
    import sys

    abs_path = os.path.abspath(data_path)
    if abs_path in _MARGINALIZER_CACHE:
        return _MARGINALIZER_CACHE[abs_path]

    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Komoltsev+2024 data file not found at {abs_path}.  "
            f"Download from Zenodo DOI 10.5281/zenodo.15407795 "
            f"(file: eos_extensions_*_pQCD-25-40.h5) and pass "
            f"its path via pqcd_config['data_path'].")

    # The Zenodo class's __init__ hardcodes the conditioned filename.
    # If we got handed a path with a different basename, the read_hdf()
    # inside __init__ would silently look for a different file in the
    # same directory and probably error; better to fail loudly here
    # with an actionable message.
    expected_basename = ('eos_extensions_s-G-1p25-0p25_l-U-1-20_'
                         'meancs2-G-0.3-0.3_pQCD-25-40.h5')
    data_dir      = os.path.dirname(abs_path)
    data_basename = os.path.basename(abs_path)
    if data_basename != expected_basename:
        raise ValueError(
            f"data_path basename must be {expected_basename!r} -- "
            f"the Zenodo eos_marginalization.py hardcodes that name "
            f"in its __init__ and opens it relative to cwd.  Got "
            f"{data_basename!r}.  If you genuinely want the "
            f"unconditioned 'prior' likelihood, that's a separate code "
            f"path; see the caveat in _get_marginalizer's docstring.")

    # Make sure eos_marginalization.py is importable.  The Zenodo
    # bundle puts it next to the data file, so add the data directory
    # to sys.path if it isn't already there.
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)

    # Construct the marginalizer.  __init__ opens the data file relative
    # to cwd, so we chdir there for the construction and only that.
    cwd_save = os.getcwd()
    try:
        os.chdir(data_dir)
        # Lazy import: deferred until the first marginalized call,
        # so the module isn't required to be installed unless used.
        import eos_marginalization as _zenodo_module
        instance = _zenodo_module.eos_marginalization()  # flag='conditioned'
        # marg_QCD_likelihood() builds the KDE table and returns the
        # closure (n0, e0, p0) -> float likelihood.  This is what we
        # cache and what _call_marginalizer invokes.
        likelihood_callable = instance.marg_QCD_likelihood()
    finally:
        os.chdir(cwd_save)

    _MARGINALIZER_CACHE[abs_path] = likelihood_callable
    return likelihood_callable


def _call_marginalizer(marginalizer, n_L, p_L, mu_L):
    """
    Thin wrapper around the Zenodo `interp_kernels` callable.

    Parameters
    ----------
    marginalizer : closure returned by _get_marginalizer.  Signature
                   (n0, e0, p0) -> float likelihood.
    n_L          : termination baryon density       [fm^-3]
    p_L          : termination pressure             [MeV/fm^3]
    mu_L         : termination chemical potential   [MeV]

    Returns
    -------
    float : log-likelihood, or -inf for points outside the
            marginalized window.

    Implementation notes
    --------------------
    Three things this wrapper handles, all of which would be silently
    wrong if elided:

      1. ENERGY-DENSITY RECONSTRUCTION.  The Zenodo callable wants
         (n0, e0, p0), but our pipeline carries (n, p, mu) at the
         termination point.  At T=0, Gibbs-Duhem gives
             eps + p = mu * n   =>   eps = mu * n - p
         exactly -- and this is literally how mu_T was computed in
         pqcd_term, so reconstruction round-trips by construction.

      2. UNIT CONVERSION.  Pipeline carries pressure in MeV/fm^3;
         Zenodo wants GeV/fm^3 for both `e0` and `p0`.  Number density
         is fm^-3 in both, so n0 passes through unchanged.

      3. LIKELIHOOD VS LOG-LIKELIHOOD.  The Zenodo function returns
         likelihood (a non-negative real); the rest of the pipeline
         uses log-likelihood.  We log-transform here, with L = 0
         mapped to -inf (out-of-window).

      4. DOMAIN GUARD.  The Zenodo `interp_kernels` _claims_ n_L is
         valid in [1, 35]*n_sat but its kernel table is built so that
         n_L >= ~34.95*n_sat actually IndexErrors due to an off-by-
         one (the bounds check uses 35*n_sat but the table indexing
         requires cnt+1 in range, which fails for the top few grid
         points).  Verified empirically against the data file:
         n_L >= ~5.59 fm^-3 (34.95*n_sat) raises IndexError; n_L > 5.6
         fm^-3 raises ValueError.  We pre-gate at 34.5*n_sat
         (= 5.52 fm^-3) for a safe margin well below both failure
         modes, and additionally catch ValueError / IndexError from
         the call so any boundary slip maps to -inf rather than
         crashing the worker.

    Reference manual outputs (with this wrapper, after log):
        e0=1.1, p0=1.0, n0=5*0.16  -> L=3.05e-13 -> log_L = -28.82
        e0=1.4, p0=0.4, n0=7*0.16  -> L=0.899    -> log_L = -0.106
        e0=1.0, p0=1.3, n0=10*0.16 -> L=8.08e-24 -> log_L = -53.17
    """
    # 1. eps reconstruction (Gibbs-Duhem, T=0)
    eps_L_MeV = mu_L * n_L - p_L                    # MeV/fm^3

    # 2. unit conversion to Zenodo's expected units
    e0_GeV    = eps_L_MeV / 1000.0                  # GeV/fm^3
    p0_GeV    = p_L / 1000.0                        # GeV/fm^3
    n0_invfm3 = float(n_L)                          # fm^-3

    # 3. domain pre-check.  n_sat = 0.16 fm^-3 in the Zenodo module.
    # Upper bound conservatively set below the off-by-one boundary
    # (see DOMAIN GUARD note above): 34.5*n_sat = 5.52 fm^-3, well
    # below the empirically-failing ~5.59 fm^-3.  Lower bound matches
    # Zenodo's stated 1*n_sat = 0.16 fm^-3.
    NSAT = 0.16
    if n0_invfm3 < 1.0 * NSAT or n0_invfm3 > 34.5 * NSAT:
        return -np.inf

    # 4. evaluate likelihood -- guard against Zenodo's known boundary
    # failure modes (ValueError = stated bounds; IndexError = off-by-
    # one near the upper limit).  Other exceptions propagate so real
    # bugs aren't silently swallowed.
    try:
        L = marginalizer(n0=n0_invfm3, e0=e0_GeV, p0=p0_GeV)
    except (ValueError, IndexError):
        return -np.inf

    L = float(L)
    if not np.isfinite(L) or L <= 0.0:
        return -np.inf
    return float(np.log(L))


# ================================================================
# Core pQCD log-likelihood at a single termination point
# ================================================================
def pqcd_log_likelihood(n_L, p_L, mu_L, marginalizer):
    """
    pQCD log-likelihood at a single termination point, evaluated
    via the Komoltsev+2024 marginalized likelihood.

    Parameters
    ----------
    n_L, p_L, mu_L : EOS point at neutron-star termination density
                     [fm^-3, MeV/fm^3, MeV]
    marginalizer   : closure returned by _get_marginalizer().
                     Required.  pqcd_term() resolves it from
                     pqcd_config["data_path"].

    Returns
    -------
    float : real-valued log-likelihood; -inf when the sample falls
            outside the marginalized window.
    """
    if marginalizer is None:
        raise ValueError(
            "pqcd_log_likelihood requires the `marginalizer` kwarg.  "
            "pqcd_term() resolves it via _get_marginalizer().")
    if not (np.isfinite(n_L) and np.isfinite(p_L) and np.isfinite(mu_L)):
        return -np.inf
    if mu_L <= 0 or n_L <= 0:
        return -np.inf
    return float(_call_marginalizer(marginalizer, n_L, p_L, mu_L))


# ================================================================
# Helpers: locate n_TOV on the stable branch
# ================================================================
def _stable_branch_last_index(M, R):
    """
    Return the index (into the ORIGINAL P_c grid) of the last stable
    star, i.e. the TOV star.  Returns -1 if no stable branch exists.

    A stable branch point has M non-decreasing along the P_c grid.
    The TOV star is the end of the stable branch (highest central P).
    """
    M = np.asarray(M)
    R = np.asarray(R)
    finite = np.isfinite(M) & np.isfinite(R)
    if not finite.any():
        return -1
    orig_idx = np.where(finite)[0]
    M_f = M[finite]
    M_cm = np.maximum.accumulate(M_f)
    stable = (M_f == M_cm)
    if not stable.any():
        return -1
    # Last stable point in the filtered array -> map back to original idx
    last_stable_filtered = np.where(stable)[0][-1]
    return int(orig_idx[last_stable_filtered])


def _rebuild_P_c_grid(guidance_config):
    """Reconstruct the central-pressure grid that exact_M_R_Lambda uses."""
    return central_pressure_grid(guidance_config)


# ================================================================
# User-facing pQCD term
# ================================================================
def pqcd_term(cs2_phys, nB_grid, M, R,
              guidance_config, pqcd_config):
    """
    Compute log_L_pQCD for one EOS sample.

    Parameters
    ----------
    cs2_phys        : (L,) physical c_s^2 curve
    nB_grid         : (L,) nB/n0 grid
    M, R            : (n_central,) from exact_M_R_Lambda, used to find n_TOV
    guidance_config : dict with eps_ref, P_ref, n_central, P_c_min, P_c_max
    pqcd_config     : dict with keys

        "data_path"      : str, REQUIRED.  Absolute or relative path
                           to the Zenodo HDF5 data file (e.g.
                           "external/zenodo_15407795/eos_extensions_..."
                           "...pQCD-25-40.h5").  Loaded lazily and
                           cached per process; see _get_marginalizer.
        "n_T_over_n0"    : fixed termination density in units of n0,
                           OR None/"n_TOV" to auto-find (default "n_TOV").
        "n0"             : fm^-3, default N0_DEFAULT.

    Returns
    -------
    log_L : float.  Real-valued log-likelihood; -inf out-of-window.
    info  : dict with n_T, p_T, mu_T, "passed".
    """
    if pqcd_config is None:
        return 0.0, {"passed": True, "reason": "pqcd disabled"}

    # ---- pull defaults
    n0 = float(pqcd_config.get("n0", N0_DEFAULT))

    # ---- reconstruct (P, eps) on the grid
    cs2 = np.asarray(cs2_phys, dtype=np.float64)
    nB  = np.asarray(nB_grid,  dtype=np.float64)
    nB_phys = nB * n0

    eps_ref = guidance_config.get("eps_ref", None)
    P_ref   = guidance_config.get("P_ref",   None)
    if eps_ref is None or P_ref is None:
        raise ValueError(
            "pqcd_term: guidance_config must carry explicit 'eps_ref' "
            "and 'P_ref' (MeV/fm^3 at the first grid point).  Load them "
            "from analysis/chEFT/cs2_BETAEQ_Lambda-500_refpoint_"
            "n0.080fm3.npz -- there is no built-in default.")

    try:
        P, eps = thermodynamic_integration_np(cs2, nB_phys, eps_ref, P_ref)
    except Exception as exc:
        return -np.inf, {"passed": False,
                         "reason": f"thermo integration failed: {exc}"}

    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(eps)):
        return -np.inf, {"passed": False,
                         "reason": "non-finite P or eps"}
    if P[-1] <= 0:
        return -np.inf, {"passed": False,
                         "reason": "negative or zero pressure at grid top"}

    # ---- choose termination density
    n_T_spec = pqcd_config.get("n_T_over_n0", "n_TOV")
    n_T_phys = None

    if n_T_spec == "n_TOV":
        # Locate the TOV star on the stable branch, then map P_c -> n_TOV
        idx_tov = _stable_branch_last_index(M, R)
        if idx_tov < 0:
            # No stable branch: fall back to top of grid
            n_T_phys = float(nB_phys[-1])
        else:
            P_c_grid = _rebuild_P_c_grid(guidance_config)
            P_c_tov  = float(P_c_grid[idx_tov])
            # Interpolate P(n_B) to find n_TOV: P is monotone in n_B for
            # a causal stable EOS.  Use numpy interp (assumes monotone P).
            if P[-1] <= P_c_tov:
                # TOV central P beyond grid -- use grid top
                n_T_phys = float(nB_phys[-1])
            else:
                n_T_phys = float(np.interp(P_c_tov, P, nB_phys))
    elif n_T_spec is None:
        n_T_phys = float(nB_phys[-1])
    else:
        # fixed value in units of n0
        n_T_phys = float(n_T_spec) * n0
        if n_T_phys > nB_phys[-1]:
            n_T_phys = float(nB_phys[-1])

    # ---- evaluate (p_T, eps_T) at n_T_phys by interpolation
    p_T   = float(np.interp(n_T_phys, nB_phys, P))
    eps_T = float(np.interp(n_T_phys, nB_phys, eps))

    if p_T <= 0 or n_T_phys <= 0:
        return -np.inf, {"passed": False,
                         "reason": "non-physical (p_T, n_T)"}

    mu_T = (eps_T + p_T) / n_T_phys

    # ---- resolve marginalizer
    data_path = pqcd_config.get("data_path", None)
    if data_path is None:
        raise ValueError(
            "pqcd_config['data_path'] is required, pointing at the "
            "Zenodo file (e.g. eos_extensions_*_pQCD-25-40.h5).  "
            "See _get_marginalizer above for the expected layout.")
    marginalizer = _get_marginalizer(data_path)

    # ---- pQCD log-likelihood
    log_L = pqcd_log_likelihood(n_T_phys, p_T, mu_T, marginalizer)

    info = {
        "n_T":    n_T_phys,
        "p_T":    p_T,
        "mu_T":   mu_T,
        "passed": np.isfinite(log_L),
    }
    return float(log_L), info