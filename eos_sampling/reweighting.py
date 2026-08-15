"""
Importance reweighting post-processor using SciPy adaptive-RK45 TOV.

For each physical c_s^2 sample, compute an exact astrophysical log-L
using scipy.integrate.solve_ivp (rtol=1e-6 by default), then convert
to self-normalised importance weights and report the effective
sample size.

The reweighter runs after sampling is complete, so no autograd is
required.  Per-sample work dispatches through joblib.Parallel by
default (n_jobs=-1); set n_jobs=1 for a serial run.

Output dict
-----------
The returned dict carries:
  weights, ESS, log_L_total
  log_L_nicer/gw/mmax        per-term log-likelihoods
  ESS_nicer/gw/mmax          per-term ESS (each data term in isolation)
  M, R, Lambda               full (N, n_central) TOV arrays, cached so
                             re-reweighting with new data is a
                             seconds-long operation
  clamp_counts               per-pulsar floor-hit counts (NICER)
When pQCD is active (pqcd_config is not None), the dict additionally
contains log_L_pqcd, pqcd_passed, pqcd_n_T, pqcd_mu_T, and ESS_pqcd.

Per-pulsar NICER mode dispatch
------------------------------
Each pulsar config dict in guidance_config["nicer_pulsars"] may carry a
"mode" key:
  "summary_gaussian" (default): existing 2D separable-Gaussian
                                likelihood from (M_obs, sigma_M, R_obs,
                                sigma_R) summary numbers.
  "kde"                       : tier-2 likelihood built from the
                                published Zenodo posterior samples,
                                with explicit divide-out of the
                                published radio mass + uniform radius
                                priors and re-addition of the radio
                                mass measurement as an independent
                                Gaussian likelihood term.  See
                                _nicer_kde_term and build_nicer_kde
                                below.
The integration scheme (line_integral vs point) is governed by the
top-level "nicer_mode" key and applies to summary-Gaussian pulsars
only; KDE pulsars always use the line-integral form (the only one
that makes sense for a 2D density that doesn't factorise in M, R).

When at least one pulsar is in mode="kde", the output dict also
carries log_L_nicer_kde (KDE contribution only), log_L_nicer_summary
(summary-Gaussian contribution only), ESS_nicer_kde and
ESS_nicer_summary.  log_L_nicer remains the SUM of the two paths so
downstream code keying on log_L_nicer is unchanged.
"""

from __future__ import annotations

import math
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, norm

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


N0_DEFAULT       = 0.16
PRESS_TO_INV_KM2 = 1.32379e-6
M_SUN_IN_KM      = 1.47663          # G M_sun / c^2 [km]

# Version stamp for the likelihood definition, written into every output
# dict so a .pt on disk states which likelihood produced it.  Bump it
# whenever the MEANING of log_L changes -- prior conventions, the
# normalisation of a marginalisation, the definition of a term -- and not
# for refactors or performance work.  Version 3 is defined by:
#   * the NICER KDE divides out each analysis's own published priors
#     (_radius_prior_log_div);
#   * the flat mass prior over the stable branch is normalised on
#     [M_MIN_POP, M_max] (_mass_prior_lognorm);
#   * weighted posterior deposits are consumed natively (build_nicer_kde);
#   * stellar structure is integrated through the SLy crust below the
#     reference point, so M, R and Lambda are whole-star quantities;
#   * samples whose mass sequence never turns over inside the nB grid
#     are rejected rather than scored on a lower bound.
LIKELIHOOD_VERSION = 3

# Lower edge of the flat prior on the mass of the observed star, used to
# NORMALISE the marginalisation over the stable branch in the NICER terms
# (see _mass_prior_lognorm).  Deliberately a fixed physical number and NOT
# M_s[0]: the branch start is set by guidance_config["P_c_min"], a pure
# numerics knob, and the likelihood must not depend on it.
M_MIN_POP = 1.0                     # M_sun

# NOTE on the thermodynamic reference point
# -----------------------------------------
# (eps_ref, P_ref) at nB_grid[0] is deliberately NOT defined in this
# module.  It must be supplied via guidance_config["eps_ref"] /
# ["P_ref"], loaded from the chEFT reference-point file
#   analysis/chEFT/cs2_BETAEQ_Lambda-500_refpoint_n0.080fm3.npz
# (see run_sampling.py).  A missing value raises immediately
# instead of silently integrating from a stale built-in default.


# ================================================================
# Crust equation of state below the reference point
# ================================================================
# Unified SLy crust in the analytical representation of Haensel &
# Potekhin (2004), A&A 428, 191: Eq. (14) with the Table 1 SLy
# coefficients for P(rho), Eq. (15) with the Table 2 SLy coefficients
# for rho(n).  rho there is the full energy density over c^2, so
# eps = rho c^2 is directly the quantity the TOV equations take.

_G_CM3_TO_MEV_FM3   = 5.60959e-13   # rho [g/cm^3] * c^2  ->  eps [MeV/fm^3]
_DYN_CM2_TO_MEV_FM3 = 6.24151e-34   # P   [dyn/cm^2]      ->  P   [MeV/fm^3]
_M0_G               = 1.66e-24      # mass per nucleon adopted by HP04 [g]

_CRUST_N_LO, _CRUST_N_HI, _CRUST_N_PTS = 6.0e-11, 0.10, 4000

_HP04_A = {1: 6.22,     2: 6.121,   3: 0.005925, 4: 0.16326,  5: 6.48,
           6: 11.4971,  7: 19.105,  8: 0.8938,   9: 6.54,    10: 11.4950,
           11: -22.775, 12: 1.5707, 13: 4.3,     14: 14.08,  15: 27.80,
           16: -1.653,  17: 1.50,   18: 14.67}

_HP04_P = {1: 0.423, 2: 2.42, 3: 0.031, 4: 0.78,
           5: 0.238, 6: 0.912, 7: 3.674}

_CRUST_CACHE: dict = {}


def _hp04_f0(x):
    """Eq. (13) of HP04, 1/(exp(x) + 1), evaluated without overflow."""
    x = np.asarray(x, dtype=np.float64)
    xp = np.clip(x, 0.0, 700.0)
    xn = np.clip(x, -700.0, 0.0)
    return np.where(x > 0.0,
                    np.exp(-xp) / (1.0 + np.exp(-xp)),
                    1.0 / (1.0 + np.exp(xn)))


def _hp04_log_P(xi):
    """Eq. (14) of HP04: log10 P [dyn/cm^2] from xi = log10 rho [g/cm^3]."""
    a = _HP04_A
    return ((a[1] + a[2] * xi + a[3] * xi ** 3) / (1.0 + a[4] * xi)
            * _hp04_f0(a[5] * (xi - a[6]))
            + (a[7] + a[8] * xi) * _hp04_f0(a[9] * (a[10] - xi))
            + (a[11] + a[12] * xi) * _hp04_f0(a[13] * (a[14] - xi))
            + (a[15] + a[16] * xi) * _hp04_f0(a[17] * (a[18] - xi)))


def _hp04_rho(n):
    """Eq. (15) of HP04: rho [g/cm^3] from baryon density n [fm^-3]."""
    p = _HP04_P
    log_n = np.log10(n)
    ratio = (1.0
             + (p[1] * n ** p[2] + p[3] * n ** p[4]) / (1.0 + p[5] * n) ** 2
             * _hp04_f0(-p[6] * (log_n + p[7]))
             + n / (8e-6 + 2.1 * n ** 0.585)
             * _hp04_f0(p[6] * (log_n + p[7])))
    return ratio * n * 1e39 * _M0_G


def crust_eos():
    """SLy crust as (nB [fm^-3], eps [MeV/fm^3], P [MeV/fm^3], c_s^2)."""
    nB = np.logspace(np.log10(_CRUST_N_LO), np.log10(_CRUST_N_HI),
                     _CRUST_N_PTS)
    rho = _hp04_rho(nB)
    xi = np.log10(rho)
    zeta = _hp04_log_P(xi)

    eps = rho * _G_CM3_TO_MEV_FM3
    P = 10.0 ** zeta * _DYN_CM2_TO_MEV_FM3

    h = 1e-5
    dzeta_dxi = (_hp04_log_P(xi + h) - _hp04_log_P(xi - h)) / (2.0 * h)
    cs2 = np.clip((P / eps) * dzeta_dxi, 1e-8, 1.0)

    return nB, eps, P, cs2


def _crust_below(P_ref_km2, eps_ref_km2):
    """Crust arrays in geometric units, truncated below the reference
    point and scaled by eps_ref / eps_SLy(P_ref) so eps(P) is continuous
    at the junction.  Returns (P, eps, cs2, scale); cached per point."""
    key = (float(P_ref_km2), float(eps_ref_km2))
    if key in _CRUST_CACHE:
        return _CRUST_CACHE[key]

    _, eps_c, P_c, cs2_c = crust_eos()
    P_c = P_c * PRESS_TO_INV_KM2
    eps_c = eps_c * PRESS_TO_INV_KM2

    if P_ref_km2 <= P_c[0] or P_ref_km2 >= P_c[-1]:
        raise ValueError(
            f"reference pressure {P_ref_km2 / PRESS_TO_INV_KM2:.4g} MeV/fm^3 "
            f"lies outside the crust table "
            f"[{P_c[0] / PRESS_TO_INV_KM2:.3g}, "
            f"{P_c[-1] / PRESS_TO_INV_KM2:.3g}] MeV/fm^3.")

    scale = eps_ref_km2 / float(np.interp(P_ref_km2, P_c, eps_c))
    if not 0.85 < scale < 1.15:
        raise ValueError(
            f"crust junction factor {scale:.3f} is far from unity.  At "
            f"nB = 0.08 fm^-3 the SLy crust gives eps = 75.9 and "
            f"P = 0.379 MeV/fm^3; check that eps_ref and P_ref are the "
            f"chEFT values at nB_grid[0] and are in MeV/fm^3.")

    keep = P_c < P_ref_km2
    out = (P_c[keep], eps_c[keep] * scale, cs2_c[keep] / scale, scale)
    _CRUST_CACHE[key] = out
    return out


def crust_junction_factor(eps_ref, P_ref):
    """eps_ref / eps_SLy(P_ref) for a reference point in MeV/fm^3.
    Raises if the reference point is unusable, which is why
    importance_weights calls it before dispatching any TOV work."""
    return _crust_below(float(P_ref) * PRESS_TO_INV_KM2,
                        float(eps_ref) * PRESS_TO_INV_KM2)[3]


def _splice_crust(P_km2, eps_km2, cs2_core):
    """Prepend the crust to a core table, returning (P, eps, cs2)."""
    P_c, eps_c, cs2_c, _ = _crust_below(P_km2[0], eps_km2[0])

    P_full = np.concatenate([P_c, P_km2])
    eps_full = np.concatenate([eps_c, eps_km2])
    cs2_full = np.concatenate([cs2_c,
                               np.asarray(cs2_core, dtype=np.float64)])

    if not np.all(np.diff(P_full) > 0.0):
        raise ValueError("spliced equation of state is not monotone in P")
    return P_full, eps_full, cs2_full


# ================================================================
# Tier-2 NICER KDE: cache and builder
# ================================================================
# Module-level cache for KDE objects built from Zenodo posterior
# files.  Keyed on (resolved abs path, bandwidth_factor, columns,
# grid, n_bins).  Lifetime is per process; under joblib n_jobs=-1,
# KDE objects are built once in the parent and shipped to workers
# via the closure (the grid interpolator is a plain object holding
# numpy arrays + scalars and pickles cleanly).

_NICER_KDE_CACHE: dict = {}


class _KDEGridInterpolator:
    """
    Fast Gaussian KDE via histogram + Gaussian-filter in pre-whitened
    coordinates.  In the bin -> 0 limit this is mathematically
    equivalent to scipy.stats.gaussian_kde with Scott's rule on the
    full empirical covariance matrix (i.e. the same KDE that the
    NICER teams used to reduce their MultiNest output to a smooth
    posterior estimator).

    Why this and not scipy.stats.gaussian_kde?
        scipy's gaussian_kde evaluates an O(n_data x n_query)
        kernel sum on every call.  For n_data ~ 4e5 (J0740) and
        n_query ~ 20 (a typical stable-branch line integral),
        that's tens of milliseconds *per sample*, which becomes
        hours of wall time in a 30k-sample re-reweight.

        The mathematics permits a much faster implementation:
          1. Whiten the data via Cholesky of the empirical
             covariance:  Y_i = L^{-1} (X_i - mean), Sigma = L L^T.
             In whitened space, scipy's bandwidth matrix
             H = h^2 * Sigma (h = n^(-1/6) for 2D Scott)
             becomes h^2 * I -- isotropic with scalar bandwidth h.
          2. In whitened space, the KDE is the convolution of the
             empirical measure with an isotropic Gaussian.  Replace
             the empirical measure with a fine 2D histogram and
             do that convolution via scipy.ndimage.gaussian_filter
             (separable, O(grid)), giving a smooth density
             estimator on a regular grid.
          3. To evaluate at an arbitrary (M, R), transform to
             whitened coordinates and bilinear-interpolate the
             grid; divide by det(L) for the Jacobian to recover
             the density in the original (M, R) space.

        Build cost: ~80 ms per pulsar (vs minutes for scipy on a
        comparable grid).  Per-call cost: ~50 us for a 20-point
        branch (vs ~100 ms for scipy gaussian_kde, ~2000x).

    Accuracy: in the bulk of the J0740 / J0437 posteriors,
    log-density agrees with scipy.stats.gaussian_kde to std
    ~ 0.02 nats and max |error| ~ 0.1 nats with default n_bins=600).
    """

    def __init__(self, samples, *, weights=None,
                 bandwidth_factor=1.0, n_bins=600):
        """
        Parameters
        ----------
        samples : (2, n) ndarray
            Posterior samples; row 0 = M, row 1 = R.
        weights : (n,) ndarray or None
            Per-sample weights, as released by analyses whose posterior
            files are NOT equal-weight (e.g. Miller+25 for J0437, whose
            Zenodo file carries a weight column).  None means equal
            weights.  Handled exactly as scipy.stats.gaussian_kde does:
            the mean and covariance become weighted, Scott's rule uses
            the effective sample size n_eff = 1 / sum(w^2) instead of n,
            and the histogram is accumulated with the weights.  Passing
            weights here is preferable to resampling the file to equal
            weight beforehand: resampling discards information (it caps
            the effective sample size at the number of draws) and adds
            multinomial noise, and it leaves a derived data file that
            has to be tracked alongside the original.
        bandwidth_factor : float
            Multiplier on Scott's rule (1.0 = scipy default).
        n_bins : int
            Number of histogram cells per axis in whitened space.
            600 keeps bandwidth/dx ~ 7 cells, well into the
            convergent regime; larger values give marginally better
            accuracy at proportional memory cost (n_bins**2 floats).
        """
        samples = np.ascontiguousarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != 2:
            raise ValueError(
                f"_KDEGridInterpolator: expected (2, n) samples, "
                f"got shape {samples.shape}")
        n = samples.shape[1]

        if weights is None:
            w = None
            n_eff = float(n)
        else:
            w = np.ascontiguousarray(weights, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError(
                    f"_KDEGridInterpolator: weights must have shape "
                    f"({n},), got {w.shape}")
            if np.any(w < 0) or not np.any(w > 0):
                raise ValueError(
                    "_KDEGridInterpolator: weights must be non-negative "
                    "with at least one positive entry.")
            w = w / w.sum()
            n_eff = float(1.0 / np.sum(w ** 2))

        if n_eff < 100:
            raise ValueError(
                f"_KDEGridInterpolator: effective sample size {n_eff:.1f} "
                f"(from {n} rows); refuse to build a KDE on so few points.")

        # Empirical mean / covariance / Cholesky for the whitening
        self._n       = int(n)
        self._n_eff   = n_eff
        self._weights = w
        if w is None:
            self._mean = samples.mean(axis=1)               # (2,)
            self._cov  = np.cov(samples)                    # (2, 2)
        else:
            self._mean = samples @ w                        # (2,)
            self._cov  = np.cov(samples, aweights=w)        # (2, 2)
        self._L      = np.linalg.cholesky(self._cov)
        self._Linv   = np.linalg.inv(self._L)
        self._det_L  = float(np.linalg.det(self._L))

        # Scott's rule bandwidth in 2D: h = n_eff^(-1/6).  With the
        # whitening above, the bandwidth matrix in whitened space
        # is h^2 * I -- isotropic with scalar bandwidth h.
        scott = n_eff ** (-1.0 / 6.0)
        self._bw       = scott * float(bandwidth_factor)
        self.factor    = self._bw                  # parity with gaussian_kde.factor

        # Whiten the samples and build a fine grid bracketing them
        # with a 5-bandwidth pad so the KDE Gaussian tail is fully
        # represented inside the grid before falling off to zero.
        Y = self._Linv @ (samples - self._mean[:, None])
        pad = 5.0 * self._bw
        x_lo, x_hi = float(Y[0].min() - pad), float(Y[0].max() + pad)
        y_lo, y_hi = float(Y[1].min() - pad), float(Y[1].max() + pad)
        self._xw_edges = np.linspace(x_lo, x_hi, n_bins + 1)
        self._yw_edges = np.linspace(y_lo, y_hi, n_bins + 1)
        self._xw_lo = float(self._xw_edges[0])
        self._yw_lo = float(self._yw_edges[0])
        self._dxw   = float(self._xw_edges[1] - self._xw_edges[0])
        self._dyw   = float(self._yw_edges[1] - self._yw_edges[0])

        # 2D histogram in whitened space, density-normalised, then
        # convolved with isotropic Gaussian of std = bandwidth.
        H, _, _ = np.histogram2d(
            Y[0], Y[1],
            bins=[self._xw_edges, self._yw_edges],
            weights=w,
            density=True)
        sigma_cells_x = self._bw / self._dxw
        sigma_cells_y = self._bw / self._dyw
        self._density_white = gaussian_filter(
            H, sigma=[sigma_cells_x, sigma_cells_y],
            mode='constant', cval=0.0)
        self._n_xw = int(self._density_white.shape[0])
        self._n_yw = int(self._density_white.shape[1])

        # Keep a reference to the raw samples so resample() can
        # match scipy.stats.gaussian_kde semantics (used only by the
        # headline-reproduction test; production runs don't call it).
        self._raw_samples = samples

    def __getstate__(self):
        """
        Drop the raw samples and weights when pickling.

        importance_weights() builds these objects in the parent and ships
        them to joblib workers through the closure.  Everything __call__
        needs is the (n_bins x n_bins) grid plus a handful of scalars --
        about 3 MB -- whereas the raw arrays of a large deposit run to
        tens of MB per worker (the Miller+25 J0437 file is ~3e6 rows, so
        ~70 MB) and are never touched during evaluation.  resample() is
        the only consumer and is test-only; it raises a clear error if
        called on an unpickled instance.
        """
        state = self.__dict__.copy()
        state["_raw_samples"] = None
        state["_weights"]     = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __call__(self, points):
        """
        Drop-in replacement for scipy.stats.gaussian_kde.__call__.

        Parameters
        ----------
        points : (2, n_query) ndarray
            (M, R) coordinates at which to evaluate the density.

        Returns
        -------
        (n_query,) ndarray of densities.  Out-of-grid points return
        0.0 (consistent with the KDE Gaussian tail being negligible
        far outside the data region; downstream code clamps anyway).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] != 2:
            raise ValueError(
                f"_KDEGridInterpolator.__call__: expected (2, n) "
                f"points, got shape {points.shape}")

        # Whiten the query points.
        Y = self._Linv @ (points - self._mean[:, None])     # (2, n_query)
        fi = (Y[0] - self._xw_lo) / self._dxw
        fj = (Y[1] - self._yw_lo) / self._dyw
        in_bounds = ((fi >= 0.0) & (fi <= self._n_xw - 1) &
                     (fj >= 0.0) & (fj <= self._n_yw - 1))
        # Clip indices for safe lookup; we'll mask out-of-bounds at end
        i0 = np.clip(fi.astype(np.int64), 0, self._n_xw - 2)
        j0 = np.clip(fj.astype(np.int64), 0, self._n_yw - 2)
        ti = fi - i0
        tj = fj - j0
        d_white = (self._density_white[i0,     j0    ] * (1.0 - ti) * (1.0 - tj) +
                   self._density_white[i0 + 1, j0    ] *        ti  * (1.0 - tj) +
                   self._density_white[i0,     j0 + 1] * (1.0 - ti) *        tj  +
                   self._density_white[i0 + 1, j0 + 1] *        ti  *        tj)
        d_white = np.where(in_bounds, d_white, 0.0)
        # Jacobian: density in original space = density_white / det(L)
        return d_white / self._det_L

    def resample(self, size, seed=None):
        """
        Match scipy.stats.gaussian_kde.resample: draw `size` samples
        from the KDE by picking a random data point and adding
        Gaussian noise with covariance bandwidth^2 * Sigma.

        Implemented by drawing isotropic noise N(0, bw^2 I) in
        whitened space, transforming back via L; the resulting
        noise has covariance L (bw^2 I) L^T = bw^2 Sigma exactly.
        """
        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)
        if self._raw_samples is None:
            raise RuntimeError(
                "_KDEGridInterpolator.resample: raw samples were dropped "
                "when this object was pickled (see __getstate__).  Rebuild "
                "the KDE in this process if you need resample().")
        if self._weights is None:
            idx = rng.integers(0, self._n, size=int(size))
        else:
            idx = rng.choice(self._n, size=int(size), p=self._weights)
        base = self._raw_samples[:, idx]                       # (2, size)
        noise_white = rng.normal(scale=self._bw,
                                  size=(2, int(size)))         # (2, size)
        noise = self._L @ noise_white
        return base + noise


def build_nicer_kde(samples_path,
                    *,
                    bandwidth_factor=1.0,
                    columns=(0, 1),
                    weights_column=None,
                    name=None,
                    grid=True,
                    n_bins=600):
    """
    Load a NICER posterior-sample file from disk and build a 2D
    Gaussian KDE over (M [M_sun], R [km]).  Cached per process on
    (abs_path, bandwidth_factor, columns, grid, n_bins).

    Parameters
    ----------
    samples_path : str or path-like
        Path to a whitespace-delimited posterior-sample file from a
        NICER Zenodo deposit.  By default the first two columns are
        treated as (M [M_sun], R [km]) -- matches Salmi+24 J0740
        ``..._mrsamples_post_equal_weights.dat`` and Choudhury+24
        J0437 ``...post_equal_weights.dat`` (the latter has 19
        columns; only the first two are mass and radius).
    bandwidth_factor : float, default 1.0
        Multiplicative factor applied AFTER Scott's-rule selection.
        1.0 reproduces scipy default; values in [1.0, 1.5] are a
        sensible widening knob for tail-sensitive analyses.
    columns : tuple[int, int], default (0, 1)
        Indices of the (mass, radius) columns in the input file.
        Order matters and is (M, R): the Amsterdam equal-weight files
        are already in that order, whereas the Miller+25 J0437 deposit
        is (R, M, weight), so it needs columns=(1, 0).
    weights_column : int or None, default None
        Index of a per-sample weight column, for deposits that are NOT
        equal-weight (Miller+25 J0437: column 2).  None treats every row
        equally, which is correct for the Amsterdam post_equal_weights
        files.  Supplying the weight column is preferable to resampling
        the file to equal weight by hand: nothing is discarded, no
        multinomial noise is added, and there is no derived data file
        to keep in sync with the original.
    name : str or None
        Human-readable label for diagnostics; defaults to the file
        basename if None.
    grid : bool, default True
        If True (default), build the fast histogram-convolution grid
        interpolator (_KDEGridInterpolator).  If False, fall back to
        scipy.stats.gaussian_kde -- mathematically identical in the
        limit of fine binning, but 2-3 orders of magnitude slower
        per call.  Use grid=False only for sanity checks against
        the reference implementation.
    n_bins : int, default 600
        Histogram resolution per axis in whitened space (only used
        when grid=True).  600 gives bandwidth/cell ~ 7 with std
        log-density error ~0.02 nats vs scipy reference; larger
        values reduce that proportionally at quadratic memory cost.

    Returns
    -------
    dict with:
        "kde"        : callable (M, R) -> density, drop-in for
                       scipy.stats.gaussian_kde.__call__; either
                       a _KDEGridInterpolator (default) or a real
                       scipy.stats.gaussian_kde (grid=False).
        "M_samples"  : (n_post,) np.float64 raw mass samples
        "R_samples"  : (n_post,) np.float64 raw radius samples
        "name"       : str
        "bandwidth"  : float, the resulting kde.factor (Scott * bw_f)
        "columns"    : tuple, echoed back for diagnostics
        "n_samples"  : int, number of posterior samples loaded
        "grid"       : bool, which path was used

    Caching
    -------
    The returned dict is the SAME object on subsequent calls with
    matching cache key; do not mutate it.  Cache is module-level and
    lives for the life of the process.
    """
    abs_path = os.path.abspath(str(samples_path))
    cache_key = (abs_path, float(bandwidth_factor),
                 tuple(int(c) for c in columns),
                 None if weights_column is None else int(weights_column),
                 bool(grid), int(n_bins))
    if cache_key in _NICER_KDE_CACHE:
        return _NICER_KDE_CACHE[cache_key]

    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"NICER samples file not found: {abs_path}.  Download from "
            f"the relevant Zenodo deposit and pass its path via the "
            f"pulsar config 'samples_path' field.")

    usecols = tuple(int(c) for c in columns)
    if weights_column is not None:
        usecols = usecols + (int(weights_column),)
    data = np.loadtxt(abs_path, usecols=usecols, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(usecols):
        raise ValueError(
            f"build_nicer_kde: expected a {len(usecols)}-column slice "
            f"from {abs_path} (cols={usecols}); got shape {data.shape}.")

    M_samples = np.ascontiguousarray(data[:, 0], dtype=np.float64)
    R_samples = np.ascontiguousarray(data[:, 1], dtype=np.float64)
    weights   = (np.ascontiguousarray(data[:, 2], dtype=np.float64)
                 if weights_column is not None else None)

    if M_samples.size < 100:
        raise ValueError(
            f"build_nicer_kde: only {M_samples.size} samples in "
            f"{abs_path}; refuse to build a KDE on so few points.")

    # Cheap orientation guard: these are always M [M_sun] and R [km],
    # so a swapped `columns` shows up immediately in the medians.
    med_M, med_R = float(np.median(M_samples)), float(np.median(R_samples))
    if not (0.5 < med_M < 3.5 and 5.0 < med_R < 20.0):
        raise ValueError(
            f"build_nicer_kde: columns={usecols[:2]} of {abs_path} give "
            f"median M={med_M:.3f} M_sun, R={med_R:.3f} km, which is not "
            f"physical.  The (mass, radius) column order is probably "
            f"swapped -- the Miller+25 J0437 deposit is (R, M, weight) "
            f"and needs columns=(1, 0).")

    points = np.vstack([M_samples, R_samples])

    if grid:
        kde = _KDEGridInterpolator(
            points,
            weights=weights,
            bandwidth_factor=float(bandwidth_factor),
            n_bins=int(n_bins))
        bw_used = kde.factor
        n_eff   = kde._n_eff
    else:
        kde = gaussian_kde(points, bw_method='scott', weights=weights)
        if abs(bandwidth_factor - 1.0) > 1e-12:
            kde.set_bandwidth(kde.factor * float(bandwidth_factor))
        bw_used = float(kde.factor)
        n_eff   = float(getattr(kde, 'neff', M_samples.size))

    obj = {
        "kde":         kde,
        "M_samples":   M_samples,
        "R_samples":   R_samples,
        "weights":     weights,
        "name":        name or os.path.basename(abs_path),
        "bandwidth":   bw_used,
        "columns":     tuple(int(c) for c in columns),
        "n_samples":   int(M_samples.size),
        "n_eff":       float(n_eff),
        "grid":        bool(grid),
    }
    _NICER_KDE_CACHE[cache_key] = obj
    return obj


def _partition_pulsars_by_mode(pulsars):
    """
    Split a list of pulsar config dicts into (summary_list, kde_list)
    by the per-pulsar "mode" field (default: "summary_gaussian").
    Validates that no unknown modes are present.
    """
    summary_list = []
    kde_list     = []
    for p in pulsars:
        m = p.get("mode", "summary_gaussian")
        if m == "summary_gaussian":
            summary_list.append(p)
        elif m == "kde":
            kde_list.append(p)
        else:
            raise ValueError(
                f"unknown NICER pulsar mode {m!r} on "
                f"{_pname(p, 0)}; expected 'summary_gaussian' or 'kde'.")
    return summary_list, kde_list


def prepare_nicer_kde_pulsars(pulsars, *, verbose=False):
    """
    Walk a list of pulsar config dicts and ensure every kde-mode
    pulsar carries a built KDE object under key "kde".  Pulsars in
    summary_gaussian mode pass through untouched.

    Returns a NEW list of dicts (shallow-copied per element); the
    input list is not mutated.  KDE objects come from build_nicer_kde
    and are therefore process-cached.

    Call this in the parent process before dispatching to joblib
    workers; gaussian_kde objects pickle cleanly through the closure
    and the workers will not have to rebuild from disk.
    """
    out = []
    for p in pulsars:
        mode = p.get("mode", "summary_gaussian")
        if mode != "kde":
            out.append(p)
            continue
        if "kde" in p and isinstance(p["kde"], dict) and "kde" in p["kde"]:
            # Already prepared (e.g. by an earlier call).  Pass through.
            out.append(p)
            continue
        samples_path     = p["samples_path"]
        bandwidth_factor = float(p.get("bandwidth_factor", 1.0))
        columns          = tuple(p.get("samples_columns", (0, 1)))
        weights_column   = p.get("weights_column", None)
        grid             = bool(p.get("kde_grid", True))     # fast by default
        n_bins           = int(p.get("kde_n_bins", 600))
        kde_obj = build_nicer_kde(
            samples_path,
            bandwidth_factor=bandwidth_factor,
            columns=columns,
            weights_column=weights_column,
            name=p.get("name"),
            grid=grid,
            n_bins=n_bins)
        if verbose:
            kind = "grid-interp" if kde_obj["grid"] else "scipy.gaussian_kde"
            wtag = ("" if weights_column is None else
                    f", weighted (n_eff={kde_obj['n_eff']:.0f})")
            print(f"  [nicer-kde] built {kind} for {kde_obj['name']}: "
                  f"{kde_obj['n_samples']} samples{wtag}, "
                  f"bandwidth={kde_obj['bandwidth']:.4f} "
                  f"(Scott*{bandwidth_factor:.2f})")
        new_p = dict(p)
        new_p["kde"] = kde_obj
        out.append(new_p)
    return out


def thermodynamic_integration_np(cs2, nB_phys, eps_ref, P_ref):
    """Heun's-method trapezoidal integration of (P, eps) from (eps_ref, P_ref).

    Uses the local spacing dn_k = nB_phys[k+1] - nB_phys[k] at each step,
    rather than caching dn = nB_phys[1] - nB_phys[0] once -- correct on
    non-uniform grids and unchanged on uniform grids.
    """
    L  = cs2.shape[0]

    P   = np.empty(L)
    eps = np.empty(L)
    P[0]   = P_ref
    eps[0] = eps_ref

    for k in range(L - 1):
        dn_k  = nB_phys[k + 1] - nB_phys[k]     # local spacing
        n_k   = nB_phys[k]
        n_kp1 = nB_phys[k + 1]

        common_k = (eps[k] + P[k]) / n_k
        dP_k     = cs2[k] * common_k
        dE_k     = common_k

        P_pred   = P[k]   + dn_k * dP_k
        eps_pred = eps[k] + dn_k * dE_k

        common_kp1 = (eps_pred + P_pred) / n_kp1
        dP_kp1     = cs2[k + 1] * common_kp1
        dE_kp1     = common_kp1

        P[k + 1]   = P[k]   + 0.5 * dn_k * (dP_k + dP_kp1)
        eps[k + 1] = eps[k] + 0.5 * dn_k * (dE_k + dE_kp1)

    return P, eps


def _build_interpolators(P_km2, eps_km2, cs2):
    def eps_of_P(P):
        return np.interp(P, P_km2, eps_km2)

    def cs2_of_P(P):
        return np.interp(P, P_km2, cs2)

    return eps_of_P, cs2_of_P


def _tov_tidal_rhs(r, y, eps_of_P, cs2_of_P, compute_tidal=True):
    P, m, yt = y
    if P <= 0.0:
        return [0.0, 0.0, 0.0]

    eps = eps_of_P(P)

    one_minus_2mr = 1.0 - 2.0 * m / r
    if one_minus_2mr <= 1e-10:
        one_minus_2mr = 1e-10

    num_P = (eps + P) * (m + 4.0 * math.pi * r**3 * P)
    den_P = r * (r - 2.0 * m)
    dP_dr = -num_P / den_P
    dm_dr = 4.0 * math.pi * r**2 * eps

    if not compute_tidal:
        return [dP_dr, dm_dr, 0.0]

    cs2 = max(cs2_of_P(P), 1e-8)
    F = (1.0 - 4.0 * math.pi * r**2 * (eps - P)) / one_minus_2mr
    Q = (4.0 * math.pi * (5.0 * eps + 9.0 * P + (eps + P) / cs2)
         / one_minus_2mr
         - 6.0 / ((r**2) * one_minus_2mr)
         - (2.0 * (m + 4.0 * math.pi * r**3 * P)
            / (r * (r - 2.0 * m)))**2)
    dy_dr = -(yt * yt + yt * F + r * r * Q) / r

    return [dP_dr, dm_dr, dy_dr]


def _surface_event(r, y, *args):
    return y[0]
_surface_event.terminal  = True
_surface_event.direction = -1


def _compute_Lambda(C, y):
    if C <= 0 or C >= 0.5:
        return 0.0
    one_2C = 1.0 - 2.0 * C
    ln_one_2C = math.log1p(-2.0 * C)
    numer = one_2C**2 * (2.0 + 2.0 * C * (y - 1.0) - y)
    D = (2.0 * C * (6.0 - 3.0 * y + 3.0 * C * (5.0 * y - 8.0))
         + 4.0 * C**3 * (13.0 - 11.0 * y
                         + C * (3.0 * y - 2.0)
                         + 2.0 * C**2 * (1.0 + y))
         + 3.0 * one_2C**2 * (2.0 - y + 2.0 * C * (y - 1.0)) * ln_one_2C)
    if abs(D) < 1e-12:
        return 0.0
    return max(0.0, (16.0 / 15.0) * numer / D)


def _solve_one_star(P_c_physical, P_km2, eps_km2, cs2, r_max=25.0,
                    rtol=1e-6, atol=1e-8, compute_tidal=True):
    eps_of_P, cs2_of_P = _build_interpolators(P_km2, eps_km2, cs2)

    r0 = 1e-3
    P_c_km2 = P_c_physical * PRESS_TO_INV_KM2
    eps_c   = eps_of_P(P_c_km2)
    m_0     = (4.0 * math.pi / 3.0) * eps_c * r0**3
    y0      = [P_c_km2, m_0, 2.0]

    sol = solve_ivp(
        _tov_tidal_rhs, (r0, r_max), y0,
        method='RK45', rtol=rtol, atol=atol,
        events=_surface_event,
        args=(eps_of_P, cs2_of_P, compute_tidal),
        dense_output=False, max_step=0.5)

    if sol.t_events[0].size > 0:
        R_km   = float(sol.t_events[0][0])
        y_R    = sol.y_events[0][0]
        M_geom = float(y_R[1])
        y_tid  = float(y_R[2])
        M_sun  = M_geom / M_SUN_IN_KM
        C      = M_geom / max(R_km, 1e-6)
        L_tid  = _compute_Lambda(C, y_tid) if compute_tidal else 0.0
        return M_sun, R_km, L_tid

    return float('nan'), float('nan'), float('nan')


def central_pressure_grid(guidance_config):
    """Log-spaced central pressures [MeV/fm^3] at which stars are solved.
    The three keys are required rather than defaulted: the grid is shared
    with the pQCD termination point, and a silent fallback would let the
    two run on different grids."""
    for key in ("n_central", "P_c_min", "P_c_max"):
        if guidance_config.get(key) is None:
            raise ValueError(
                f"guidance_config must carry explicit '{key}'.  The "
                f"central-pressure grid is shared with the pQCD termination "
                f"point and is deliberately not defaulted.")
    return np.logspace(np.log10(float(guidance_config["P_c_min"])),
                       np.log10(float(guidance_config["P_c_max"])),
                       int(guidance_config["n_central"]))


def exact_M_R_Lambda(cs2_phys, nB_grid, guidance_config,
                     rtol=1e-6, atol=1e-8):
    """
    Low-level entry point: run the SciPy solver for a single EOS.
    Returns (M, R, Lambda, truncated); M, R, Lambda are 1-D numpy arrays
    across the P_c grid, truncated flags a mass sequence still rising at
    the last solved point (nanmax(M) is then only a lower bound).
    """
    n0 = float(guidance_config.get("n0", N0_DEFAULT))
    nB_phys = np.asarray(nB_grid, dtype=np.float64) * n0

    eps_ref = guidance_config.get("eps_ref", None)
    P_ref   = guidance_config.get("P_ref",   None)
    if eps_ref is None or P_ref is None:
        raise ValueError(
            "exact_M_R_Lambda: guidance_config must carry explicit "
            "'eps_ref' and 'P_ref' (MeV/fm^3 at nB_grid[0]).  Load them "
            "from analysis/chEFT/cs2_BETAEQ_Lambda-500_refpoint_"
            "n0.080fm3.npz -- there is no built-in default.")

    P, eps = thermodynamic_integration_np(
        np.asarray(cs2_phys, dtype=np.float64), nB_phys, eps_ref, P_ref)

    P_km2, eps_km2, cs2_km2 = _splice_crust(
        P * PRESS_TO_INV_KM2,
        eps * PRESS_TO_INV_KM2,
        cs2_phys)

    P_c_grid  = central_pressure_grid(guidance_config)
    n_central = P_c_grid.size
    r_max     = float(guidance_config.get("r_max",    25.0))
    compute_tidal = (guidance_config.get("gw") is not None)

    M = np.full(n_central, np.nan)
    R = np.full(n_central, np.nan)
    L = np.full(n_central, np.nan)

    for i, P_c in enumerate(P_c_grid):
        if P_c > P[-1]:
            break          # beyond the EOS table; M, R, L stay NaN
        try:
            Mi, Ri, Li = _solve_one_star(
                P_c, P_km2, eps_km2, cs2_km2,
                r_max=r_max, rtol=rtol, atol=atol,
                compute_tidal=compute_tidal)
        except Exception:
            Mi = Ri = Li = float('nan')
        M[i] = Mi
        R[i] = Ri
        L[i] = Li

    finite = np.isfinite(M)
    truncated = bool(finite.any()
                     and int(np.nanargmax(M)) == int(np.where(finite)[0][-1]))

    return M, R, L, truncated


def _stable_branch(M, R, L):
    """
    Keep ONLY points where M equals the running cummax, i.e. strict
    monotone-non-decreasing.  An older tolerance (M >= cummax - 1e-6)
    occasionally kept a point just after turnover on a numerically flat
    shoulder and contaminated M_max; strict equality is conservative
    and drops such points.
    """
    finite = np.isfinite(M) & np.isfinite(R)
    if not finite.any():
        return np.array([]), np.array([]), np.array([])
    M_f = M[finite]
    R_f = R[finite]
    L_f = L[finite]
    M_cm = np.maximum.accumulate(M_f)
    stable = (M_f == M_cm)          # strict equality
    return M_f[stable], R_f[stable], L_f[stable]


def interp_on_stable_branch(M_row, Y_row, M_target):
    """Y (R or Lambda) at M_target on one sample's stable branch.
    Returns NaN when the branch has fewer than two points or does not
    span M_target."""
    M_row = np.asarray(M_row, dtype=np.float64)
    Y_row = np.asarray(Y_row, dtype=np.float64)
    M_s, Y_s, _ = _stable_branch(M_row, Y_row, Y_row)
    if M_s.size < 2:
        return float("nan")
    if (M_target < M_s.min()) or (M_target > M_s.max()):
        return float("nan")
    return float(np.interp(M_target, M_s, Y_s))


def per_sample_fiducials(M, R, Lambda, target_masses=(1.4, 2.08),
                         M_max_pred=None):
    """R(M_t) and Lambda(M_t) per sample, plus M_TOV.
    M, R, Lambda are the (N, n_central) cached TOV arrays; M_max_pred,
    when given, is passed through instead of recomputing nanmax(M).
    Returns a dict of (N,) arrays keyed "R(1.4)", "Lambda(1.4)", ...,
    "M_TOV"."""
    M      = np.asarray(M,      dtype=np.float64)
    R      = np.asarray(R,      dtype=np.float64)
    Lambda = np.asarray(Lambda, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be (N, n_central); got {M.shape}")
    N = M.shape[0]

    out = {}
    for Mt in target_masses:
        out[f"R({Mt})"] = np.array(
            [interp_on_stable_branch(M[n], R[n], Mt) for n in range(N)])
        out[f"Lambda({Mt})"] = np.array(
            [interp_on_stable_branch(M[n], Lambda[n], Mt) for n in range(N)])

    if M_max_pred is not None:
        out["M_TOV"] = np.asarray(M_max_pred, dtype=np.float64).copy()
    else:
        any_finite = np.isfinite(M).any(axis=1)
        M_TOV = np.full(N, np.nan)
        if any_finite.any():
            M_TOV[any_finite] = np.nanmax(M[any_finite], axis=1)
        out["M_TOV"] = M_TOV
    return out


def _pname(p, i):
    """Pulsar display name, falling back to index if no name is given."""
    return p.get("name", f"pulsar_{i}")


_NEG_INF = -1.0e30


def _mass_prior_lognorm(M_s):
    """
    log of the normalisation of the flat mass prior on the stable branch.

    The NICER marginalisation is

        L_p(EOS) = int dM  p(M | EOS)  f_p(M, R(M)) ,

    with p(M | EOS) flat and NORMALISED on [M_MIN_POP, M_max(EOS)], i.e.
    p = 1 / (M_max - M_MIN_POP).  Dropping this factor (integrating dM
    bare) multiplies every pulsar's likelihood by (M_max - M_MIN_POP) and
    therefore imposes an unintended preference for equations of state with
    long stable branches -- with three pulsars that is a factor of up to
    ~3-4 across a realistic M_max range.  Returns log(M_max - M_MIN_POP),
    to be SUBTRACTED from each per-pulsar log-L.

    A branch that does not reach M_MIN_POP has no prior support at all;
    we return a large positive number so the caller's subtraction floors
    the pulsar (such a branch cannot host any of the NICER pulsars anyway).
    """
    M_max = float(M_s[-1])
    span  = M_max - M_MIN_POP
    if span <= 0.0:
        return 1.0e6
    return math.log(span)


def _radius_prior_log_div(M_s, R_s, rp):
    """
    Divide out the published radius prior: returns
    (-log pi_R(R | M), in_support) evaluated on the stable branch.

    Supported forms
    ---------------
    {"type": "uniform_km", "lo": <float km>, "hi": <float km>,
     "compactness_lo": <float or None>}
        Flat in R between two edges in km.  This is the X-PSI convention
        used by the Amsterdam analyses (Salmi+24 J0740, Choudhury+24
        J0437): flat between 3 r_g(1 M_sun) = 4.43 km and 16 km, further
        restricted by the compactness condition R_pol / r_g(M) > 3.
        Setting "compactness_lo": 3.0 raises the lower edge to 3 r_g(M)
        where that is the binding constraint, which is what makes the
        conditional normalisation depend on M.
        (X-PSI applies the condition to the POLAR radius; using the
        equatorial radius here is conservative and, at the compactness
        of these stars, a sub-percent difference.)

    {"type": "uniform_compactness", "lo_over_rg": a, "hi_over_rg": b}
        Flat in R between a*r_g(M) and b*r_g(M), r_g(M) = G M / c^2.
        This is the Maryland convention: Miller+25 sample flat in the
        inverse compactness c^2 R_e /(GM) over [3.2, 8.0], so at fixed M
        the radius is flat over [3.2 r_g, 8.0 r_g] and the conditional
        normalisation is 1 / (4.8 r_g(M)), i.e. proportional to 1/M.

    """
    r_g = M_SUN_IN_KM * M_s                      # G M / c^2 [km]
    rtype = rp.get("type", "uniform_km")

    if rtype == "uniform_km":
        R_lo = float(rp.get("lo", 4.4)) * np.ones_like(M_s)
        c_lo = rp.get("compactness_lo", None)
        if c_lo is not None:
            R_lo = np.maximum(R_lo, float(c_lo) * r_g)
        R_hi = float(rp["hi"]) * np.ones_like(M_s)

    elif rtype == "uniform_compactness":
        R_lo = float(rp["lo_over_rg"]) * r_g
        R_hi = float(rp["hi_over_rg"]) * r_g

    else:
        raise ValueError(
            f"_radius_prior_log_div: unknown radius_prior type {rtype!r}; "
            f"expected 'uniform_km' or 'uniform_compactness'.")

    width      = R_hi - R_lo
    in_support = (R_s >= R_lo) & (R_s <= R_hi) & (width > 0)
    log_div    = np.where(in_support,
                          np.log(np.maximum(width, 1e-30)),
                          _NEG_INF)
    return log_div, in_support


def _nicer_term_line_integral(M_s, R_s, pulsars):
    """
    Line-integral NICER likelihood.  Also returns a dict
    {pulsar_name: 0 or 1} flagging whether the -1e6 floor fired for
    this sample (used by importance_weights to aggregate per-pulsar
    clamp counts).
    """
    clamp_hits = {}

    if M_s.size < 2:
        for i, p in enumerate(pulsars):
            clamp_hits[_pname(p, i)] = 1
        return -1.0e6 * len(pulsars), clamp_hits

    NEG_INF = -1.0e30
    log_L_total = 0.0
    dM = np.diff(M_s)
    valid = dM > 0
    if not valid.any():
        for i, p in enumerate(pulsars):
            clamp_hits[_pname(p, i)] = 1
        return -1.0e6 * len(pulsars), clamp_hits

    log_dM = np.log(np.where(valid, dM, 1e-30))

    # Flat mass prior, normalised on [M_MIN_POP, M_max]; see
    # _mass_prior_lognorm.  Points below M_MIN_POP carry no prior support.
    log_mass_norm = _mass_prior_lognorm(M_s)
    below_min     = (M_s < M_MIN_POP)

    for i, p in enumerate(pulsars):
        M_obs, sig_M = float(p["M_obs"]), float(p["sigma_M"])
        R_obs, sig_R = float(p["R_obs"]), float(p["sigma_R"])
        log_f = (-0.5 * ((M_s - M_obs) / sig_M) ** 2
                 -0.5 * ((R_s - R_obs) / sig_R) ** 2)
        log_f = np.where(below_min, NEG_INF, log_f)
        log_sum_lr = np.logaddexp(log_f[:-1], log_f[1:])
        log_seg = log_sum_lr - math.log(2.0) + log_dM
        log_seg = np.where(valid, log_seg, NEG_INF)
        lL = float(np.logaddexp.reduce(log_seg)) - log_mass_norm
        clamp_hits[_pname(p, i)] = int(lL < -1.0e6)   # floor-hit flag
        log_L_total += max(lL, -1.0e6)
    return log_L_total, clamp_hits


def _nicer_term_point(M_s, R_s, pulsars):
    """
    Point-evaluation NICER likelihood.  Returns an API-consistent
    clamp_hits dict; point mode doesn't clamp in the interior but
    does flag the degenerate-branch case.
    """
    clamp_hits = {_pname(p, i): 0 for i, p in enumerate(pulsars)}
    if M_s.size < 2:
        for k in clamp_hits:
            clamp_hits[k] = 1
        return -1.0e6 * len(pulsars), clamp_hits
    log_L_total = 0.0
    for i, p in enumerate(pulsars):
        M_obs = float(p["M_obs"])
        R_obs = float(p["R_obs"])
        sig_R = float(p["sigma_R"])
        R_pred = float(np.interp(M_obs, M_s, R_s))
        log_L_total += -0.5 * ((R_pred - R_obs) / sig_R) ** 2
    return log_L_total, clamp_hits


def _nicer_kde_term(M_s, R_s, pulsars):
    """
    Tier-2 NICER likelihood: line integral over the stable branch of
    a KDE-based per-point density that explicitly divides out the
    published NICER priors and re-adds the radio mass measurement
    as an independent Gaussian likelihood.

    Per-point density (in log space) along the stable branch:

        log f(M, R) =   log KDE(M, R)                       (data fit)
                      - log pi_M(M)                         (divide out
                                                             radio prior)
                      - log pi_R(R | M)                     (divide out
                                                             uniform R prior)
                      + log L_radio(M)                      (re-add radio
                                                             likelihood as
                                                             independent
                                                             data)

    For Gaussian pi_M = N(M | M_p, sigma_p) and Gaussian
    L_radio = N(M | M_l, sigma_l), the explicit terms reduce to:

        - log pi_M(M)   = +0.5 * ((M - M_p) / sigma_p) ** 2
                          + (M-independent constant we drop)

        + log L_radio(M) = -0.5 * ((M - M_l) / sigma_l) ** 2
                          + (M-independent constant we drop)

    For uniform pi_R(R | M) on [R_lo(M), R_hi(M)]:

        - log pi_R(R | M) = + log(R_hi(M) - R_lo(M))         (R in support)
        - log pi_R(R | M) = -inf                             (R out of
                                                              support)

    The edges follow whichever analysis produced the samples; see
    _radius_prior_log_div for the two published conventions (X-PSI /
    Amsterdam: flat in R on [4.4, 16] km with a compactness cut;
    Maryland / Miller+25: flat in inverse compactness on [3.2, 8.0]).
    The whole integral is divided by (M_max - M_MIN_POP) to normalise
    the flat mass prior over the stable branch (_mass_prior_lognorm).

    When sigma_p == sigma_l and
    M_p == M_l (the default for divide-and-re-add of the same
    radio measurement), the Gaussian terms cancel exactly and only
    the log KDE term and the M-dependent radius-prior normalisation
    remain -- but we keep them written separately for ablation
    (e.g. dropping the radio likelihood, or testing a wider re-add
    sigma to weaken the radio constraint).

    Trapezoidal line integral over the stable branch in M, then a
    -1e6 floor on the per-pulsar log-L if the integral underflows
    (no overlap with the published support) -- exactly the same
    contract as _nicer_term_line_integral.

    Parameters
    ----------
    M_s, R_s : 1-D arrays
        Stable-branch (M [M_sun], R [km]) from _stable_branch.
    pulsars  : list of dicts
        Each dict must carry:
          "name"            : str (display)
          "kde"             : object returned by build_nicer_kde
                              (with key "kde" -> gaussian_kde)
          "mass_prior"      : {"type": "gaussian", "M": float,
                                                   "sigma": float}
          "radius_prior"    : {"type": "uniform",
                               "lo": "R_Schw" or float,
                               "hi": float}
          "mass_likelihood" : {"type": "gaussian", "M": float,
                                                   "sigma": float}

    Returns
    -------
    log_L_total : float
        Sum across pulsars of per-pulsar line-integral log-L
        (each with -1e6 floor).
    clamp_hits  : dict {pulsar_name: 0|1}
        1 indicates the -1e6 floor fired for that pulsar
        on this sample.
    """
    clamp_hits = {}

    if M_s.size < 2:
        for i, p in enumerate(pulsars):
            clamp_hits[_pname(p, i)] = 1
        return -1.0e6 * len(pulsars), clamp_hits

    NEG_INF = -1.0e30
    log_L_total = 0.0
    dM = np.diff(M_s)
    valid = dM > 0
    if not valid.any():
        for i, p in enumerate(pulsars):
            clamp_hits[_pname(p, i)] = 1
        return -1.0e6 * len(pulsars), clamp_hits

    log_dM = np.log(np.where(valid, dM, 1e-30))

    # Flat mass prior, normalised on [M_MIN_POP, M_max]; see
    # _mass_prior_lognorm.  Points below M_MIN_POP carry no prior support.
    log_mass_norm = _mass_prior_lognorm(M_s)
    below_min     = (M_s < M_MIN_POP)

    # Stack (M, R) once for vectorised KDE evaluation -- we hand this
    # to each pulsar's gaussian_kde in a single call (the inner kernel
    # sum is what dominates KDE cost, so per-pulsar vectorisation
    # over n_stable points is essential).
    points = np.vstack([M_s, R_s])               # (2, n_stable)

    for i, p in enumerate(pulsars):
        # --- unpack & validate config ---
        if "kde" not in p:
            raise ValueError(
                f"_nicer_kde_term: pulsar {_pname(p, i)!r} has no "
                f"'kde' key.  Did you forget to call "
                f"prepare_nicer_kde_pulsars()?")
        kde_obj = p["kde"]
        kde     = kde_obj["kde"]

        mp = p["mass_prior"]
        rp = p["radius_prior"]
        ml = p["mass_likelihood"]
        if mp.get("type") != "gaussian":
            raise ValueError(
                f"_nicer_kde_term: only gaussian mass_prior is "
                f"supported, got {mp.get('type')!r} on "
                f"{_pname(p, i)}.")
        if rp.get("type") not in ("uniform_km", "uniform_compactness"):
            raise ValueError(
                f"_nicer_kde_term: radius_prior type must be "
                f"'uniform_km' or 'uniform_compactness', got "
                f"{rp.get('type')!r} on {_pname(p, i)}.")
        if ml.get("type") != "gaussian":
            raise ValueError(
                f"_nicer_kde_term: only gaussian mass_likelihood is "
                f"supported, got {ml.get('type')!r} on "
                f"{_pname(p, i)}.")

        M_p   = float(mp["M"]);     sig_p = float(mp["sigma"])
        M_l   = float(ml["M"]);     sig_l = float(ml["sigma"])

        # --- KDE evaluation, vectorised over the n_stable branch points ---
        kde_vals = kde(points)                    # (n_stable,)
        # tiny floor on the KDE value so np.log doesn't go to -inf
        # for points way out in the tail; the in_support mask below
        # is what really kills out-of-prior-support points.
        log_kde = np.log(np.maximum(kde_vals, 1e-300))

        # --- divide out radio mass prior (Gaussian; M-dependent only) ---
        log_prior_div_M = 0.5 * ((M_s - M_p) / sig_p) ** 2

        # --- divide out the published radius prior ---
        log_prior_div_R, in_support = _radius_prior_log_div(M_s, R_s, rp)

        # --- re-add radio likelihood (Gaussian; independent data) ---
        log_radio_lik = -0.5 * ((M_s - M_l) / sig_l) ** 2

        # --- per-point log-density along the stable branch ---
        log_f = log_kde + log_prior_div_M + log_prior_div_R + log_radio_lik
        log_f = np.where(in_support & ~below_min, log_f, NEG_INF)

        # --- trapezoidal line integral, identical scheme to
        #     _nicer_term_line_integral; segment with one in-support
        #     and one out-of-support endpoint degrades naturally to
        #     half-segment via logaddexp(finite, NEG_INF) = finite ---
        log_sum_lr = np.logaddexp(log_f[:-1], log_f[1:])
        log_seg    = log_sum_lr - math.log(2.0) + log_dM
        log_seg    = np.where(valid, log_seg, NEG_INF)
        lL = float(np.logaddexp.reduce(log_seg)) - log_mass_norm

        clamp_hits[_pname(p, i)] = int(lL < -1.0e6)
        log_L_total += max(lL, -1.0e6)

    return log_L_total, clamp_hits


def _gw_term(M_s, L_s, gw):
    if gw is None:
        return 0.0
    if M_s.size < 2:
        return -1.0e6
    m1, m2 = float(gw["m1"]), float(gw["m2"])
    if M_s[0] > min(m1, m2) or M_s[-1] < max(m1, m2):
        return -1.0e6
    L1 = float(np.interp(m1, M_s, L_s))
    L2 = float(np.interp(m2, M_s, L_s))
    num = ((m1 + 12.0 * m2) * m1**4 * L1
           + (m2 + 12.0 * m1) * m2**4 * L2)
    den = (m1 + m2)**5
    tilde_L_pred = (16.0 / 13.0) * num / den
    delta      = tilde_L_pred - float(gw["Lambda_tilde_obs"])
    half_width = float(gw["sigma_plus"]) if delta > 0 else float(gw["sigma_minus"])

    # `half_width` is a credible-interval half-width; the Gaussian below
    # needs a 1-sigma scale.  If the half-widths were quoted at credible
    # level CL (e.g. the LVC 90% HPD interval 300 +420/-230, Abbott+19,
    # 1805.11579), convert: for a Gaussian the half-width is z*sigma with
    # z = Phi^{-1}((1+CL)/2).  Absent/None => treated as already 1-sigma.
    cl    = gw.get("credible_level", None)
    z     = 1.0 if cl is None else float(norm.ppf(0.5 * (1.0 + float(cl))))
    sigma = half_width / z

    return -0.5 * (delta / sigma) ** 2


def _mmax_term(M, mmax):
    if mmax is None:
        return 0.0
    finite = np.isfinite(M)
    if not finite.any():
        return -1.0e6
    M_max = float(np.nanmax(M))
    M_lower = float(mmax["M_lower_bound"])
    sigma   = float(mmax["sigma"])
    shortfall = min(M_max - M_lower, 0.0)
    return -0.5 * (shortfall / sigma) ** 2


def _pqcd_enabled(pqcd_config):
    """Single source of truth for whether pQCD is active."""
    return pqcd_config is not None


def exact_log_likelihood(cs2_phys, nB_grid, guidance_config,
                         nicer_mode="line_integral",
                         rtol=1e-6, atol=1e-8,
                         pqcd_config=None):
    """
    Compute the exact astrophysical log-L for a single EOS using SciPy.

    Returns (log_L_total, info), where info carries:
      M, R, Lambda             full TOV arrays (cached for re-reweighting)
      log_L_nicer/gw/mmax      per-term log-likelihoods
      clamp_hits               per-pulsar floor indicator
      M_max_pred, n_stable
      log_L_pqcd, pqcd_passed, pqcd_n_T, pqcd_mu_T  (only if
          pqcd_config is active; otherwise absent)
    """
    M, R, L, truncated = exact_M_R_Lambda(cs2_phys, nB_grid, guidance_config,
                                          rtol=rtol, atol=atol)
    M_s, R_s, L_s = _stable_branch(M, R, L)

    pulsars = guidance_config.get("nicer_pulsars", []) or []
    gw      = guidance_config.get("gw",            None)
    mmax    = guidance_config.get("mmax",          None)

    # ---- NICER: per-pulsar mode dispatch ----
    # summary_gaussian pulsars go through the existing line_integral /
    # point path; kde pulsars go through _nicer_kde_term.  Both
    # contribute to the same log_L_nicer total but we also expose the
    # split in the info dict (log_L_nicer_summary / log_L_nicer_kde)
    # for diagnostics.
    nic_summary = 0.0
    nic_kde     = 0.0
    clamp_hits  = {}
    if pulsars:
        summary_pulsars, kde_pulsars = _partition_pulsars_by_mode(pulsars)

        if summary_pulsars:
            if nicer_mode == "line_integral":
                nic_g, ch_g = _nicer_term_line_integral(
                    M_s, R_s, summary_pulsars)
            elif nicer_mode == "point":
                nic_g, ch_g = _nicer_term_point(
                    M_s, R_s, summary_pulsars)
            else:
                raise ValueError(f"unknown nicer_mode {nicer_mode!r}")
            nic_summary = nic_g
            clamp_hits.update(ch_g)

        if kde_pulsars:
            nic_k, ch_k = _nicer_kde_term(M_s, R_s, kde_pulsars)
            nic_kde = nic_k
            clamp_hits.update(ch_k)

    nic = nic_summary + nic_kde

    gw_ll   = _gw_term(M_s, L_s, gw)
    mmax_ll = _mmax_term(M, mmax)

    log_L_total = nic + gw_ll + mmax_ll
    info = {
        "M_max_pred":  (float(np.nanmax(M))
                        if np.any(np.isfinite(M)) else float('nan')),
        "n_stable":    int(M_s.size),
        "truncated":   bool(truncated),
        "log_L_nicer":         float(nic),
        "log_L_nicer_summary": float(nic_summary),
        "log_L_nicer_kde":     float(nic_kde),
        "log_L_gw":    float(gw_ll),
        "log_L_mmax":  float(mmax_ll),
        "clamp_hits":  clamp_hits,
        "M":           M,        # cached for later re-reweighting
        "R":           R,
        "Lambda":      L,
    }

    # ---- Optional pQCD term ----------------------------------------
    if _pqcd_enabled(pqcd_config):
        # Lazy import avoids a circular dependency at package load
        # (pqcd.py imports from this module).
        from .pqcd import pqcd_term
        log_L_pqcd_val, pq_info = pqcd_term(
            cs2_phys, nB_grid, M, R,
            guidance_config=guidance_config,
            pqcd_config=pqcd_config)
        log_L_total = log_L_total + log_L_pqcd_val
        info["log_L_pqcd"]  = float(log_L_pqcd_val)
        info["pqcd_passed"] = bool(pq_info.get("passed", False))
        info["pqcd_n_T"]    = float(pq_info.get("n_T",  np.nan))
        info["pqcd_mu_T"]   = float(pq_info.get("mu_T", np.nan))

    return float(log_L_total), info


# Module-level worker so joblib can pickle it across processes.
def _one_sample_worker(cs2_sample, nB_grid_np, guidance_config,
                       nicer_mode, rtol, atol, pqcd_config):
    try:
        lL, info = exact_log_likelihood(
            cs2_sample, nB_grid_np, guidance_config,
            nicer_mode=nicer_mode, rtol=rtol, atol=atol,
            pqcd_config=pqcd_config)
        return lL, info, None
    except Exception as exc:
        return -np.inf, None, repr(exc)


def _ess_from_logw(log_w, N, context=""):
    """Turn log-weights into normalized weights + ESS.
    An all -inf log-weight array is a failed run, not a flat posterior:
    returning uniform weights there reports ESS == N for a run in which
    nothing was computed.  Raise instead."""
    where = f" [{context}]" if context else ""
    finite = np.isfinite(log_w)
    if not finite.any():
        raise RuntimeError(
            f"every log-weight is -inf{where}: this is a failed run, not a "
            f"flat posterior.  Nothing is normalisable, so no weights and "
            f"no ESS are defined.")
    m = log_w[finite].max()
    shifted = np.where(finite, log_w - m, -np.inf)
    raw = np.exp(shifted)
    Z = raw.sum()
    if Z <= 0:
        raise RuntimeError(
            f"log-weights underflowed to zero total mass{where} even after "
            f"shifting by the maximum; the likelihood cannot be normalised.")
    w = raw / Z
    ess = 1.0 / np.sum(w * w)
    return w, float(ess)


def importance_weights(samples_phys, nB_grid, guidance_config,
                       nicer_mode="line_integral",
                       rtol=1e-6, atol=1e-8, verbose=True,
                       n_jobs=-1,
                       pqcd_config=None):
    """
    Self-normalised importance weights from exact SciPy log-L.

    Parameters
    ----------
    n_jobs : int
        joblib Parallel n_jobs.  -1 uses all cores, 1 forces serial
        (for debugging).  Falls back to serial if joblib is not
        installed.
    pqcd_config : dict or None
        Optional Komoltsev-Kurkela 2022 pQCD hard-cut config.
        See eos_sampling.pqcd for keys.  None or {"mode": "none"}
        disables pQCD -- output matches pre-pQCD behaviour exactly.

    Returns
    -------
      log_L_exact, log_weights, weights, ESS, N, M_max_pred, n_stable
      log_L_nicer, log_L_gw, log_L_mmax
      ESS_nicer,   ESS_gw,   ESS_mmax
      M, R, Lambda    (N, n_central) arrays
      clamp_counts    dict {pulsar_name: n_samples_clamped}
      log_L_pqcd, pqcd_passed, pqcd_n_T, pqcd_mu_T, ESS_pqcd
        (only if pqcd_config is active; absent otherwise)
    """
    if (guidance_config.get("eps_ref") is None
            or guidance_config.get("P_ref") is None):
        raise ValueError(
            "importance_weights: guidance_config must carry explicit "
            "'eps_ref' and 'P_ref' (MeV/fm^3 at the first grid point).  "
            "Load them from analysis/chEFT/cs2_BETAEQ_Lambda-500_"
            "refpoint_n0.080fm3.npz.  Failing here, before any TOV "
            "work, on purpose.")
    try:
        import torch
        if isinstance(samples_phys, torch.Tensor):
            samples_phys = samples_phys.detach().cpu().numpy()
        if isinstance(nB_grid, torch.Tensor):
            nB_grid = nB_grid.detach().cpu().numpy()
    except ImportError:
        pass

    samples_phys = np.asarray(samples_phys, dtype=np.float64)
    nB_grid      = np.asarray(nB_grid,      dtype=np.float64)

    if samples_phys.ndim != 2:
        raise ValueError(f"samples_phys must be 2D, got shape {samples_phys.shape}")
    if samples_phys.shape[1] != nB_grid.shape[0]:
        raise ValueError(
            f"samples_phys columns ({samples_phys.shape[1]}) must match "
            f"nB_grid length ({nB_grid.shape[0]})")

    N = samples_phys.shape[0]
    pqcd_on = _pqcd_enabled(pqcd_config)
    use_parallel = _HAS_JOBLIB and n_jobs != 1 and N > 1
    backend_str  = "joblib-parallel" if use_parallel else "serial"

    # ---- Pre-build any KDE-mode pulsar objects in the parent process
    # so joblib workers receive them via closure (gaussian_kde pickles
    # cleanly).  This avoids each worker reloading the Zenodo file
    # from disk.  Pulsars in summary_gaussian mode pass through.
    pulsars_in = guidance_config.get("nicer_pulsars", []) or []
    if any(p.get("mode", "summary_gaussian") == "kde" for p in pulsars_in):
        prepared = prepare_nicer_kde_pulsars(pulsars_in, verbose=verbose)
        # Clone the config so we don't mutate the caller's dict.
        guidance_config = dict(guidance_config)
        guidance_config["nicer_pulsars"] = prepared

    # Fail on an unusable crust junction here, not inside a joblib
    # worker where the exception would silently zero every weight.
    junction = crust_junction_factor(guidance_config["eps_ref"],
                                     guidance_config["P_ref"])

    if verbose:
        extra = ", pqcd=kk2024_marginalized" if pqcd_on else ""
        print(f"  [reweighting] Running SciPy TOV on {N} samples "
              f"(rtol={rtol:.0e}, atol={atol:.0e}, nicer_mode={nicer_mode}, "
              f"backend={backend_str}, n_jobs={n_jobs}{extra})")
        print(f"  [reweighting] SLy crust joined at P_ref = "
              f"{float(guidance_config['P_ref']):.4f} MeV/fm^3; "
              f"eps continuity factor {junction:.4f}")

    # Parallel dispatch.
    # We disable joblib's built-in progress meter (verbose=0) because its
    # "[Parallel(n_jobs=-1)]: Done X tasks ..." lines fire on a heuristic
    # geometric schedule that's hard to read.  Instead we use
    # return_as="generator" (joblib >= 1.3) to consume results as they
    # arrive and print our own counter in fixed steps, overwriting the
    # previous line in place via \r.
    if use_parallel:
        # Step size for progress output: print every step_progress
        # completions, but never more often than every 5% of the run.
        step_progress = max(1, min(50, N // 20))
        if verbose:
            print(f"  [reweighting] progress (parallel, "
                  f"reporting every {step_progress} samples):")
        try:
            gen = Parallel(n_jobs=n_jobs, verbose=0,
                           return_as="generator")(
                delayed(_one_sample_worker)(
                    samples_phys[n], nB_grid, guidance_config,
                    nicer_mode, rtol, atol, pqcd_config)
                for n in range(N))
            results = []
            for n, r in enumerate(gen, start=1):
                results.append(r)
                if verbose and (n % step_progress == 0 or n == N):
                    print(f"    {n:>5d}/{N} samples done",
                          end="\r", flush=True)
            if verbose:
                print()    # drop newline so subsequent output starts fresh
        except TypeError:
            # joblib < 1.3 (no return_as kwarg): fall back to silent
            # parallel call without progress tracking.
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_one_sample_worker)(
                    samples_phys[n], nB_grid, guidance_config,
                    nicer_mode, rtol, atol, pqcd_config)
                for n in range(N))
            if verbose:
                print(f"    {N}/{N} samples done "
                      f"(joblib < 1.3: progress not available)",
                      flush=True)
    else:
        results = []
        step_progress = max(1, min(50, N // 20))
        for n in range(N):
            results.append(_one_sample_worker(
                samples_phys[n], nB_grid, guidance_config,
                nicer_mode, rtol, atol, pqcd_config))
            if verbose and ((n + 1) % step_progress == 0
                            or (n + 1) == N):
                print(f"    {n+1:>5d}/{N} samples done",
                      end="\r", flush=True)
        if verbose:
            print()    # drop newline so subsequent output starts fresh

    # Unpack
    log_L_exact         = np.full(N, -np.inf, dtype=np.float64)
    log_L_nicer         = np.full(N, -np.inf, dtype=np.float64)
    log_L_nicer_summary = np.full(N, -np.inf, dtype=np.float64)
    log_L_nicer_kde     = np.full(N, -np.inf, dtype=np.float64)
    log_L_gw            = np.full(N, -np.inf, dtype=np.float64)
    log_L_mmax          = np.full(N, -np.inf, dtype=np.float64)
    M_max_arr    = np.full(N, np.nan,  dtype=np.float64)
    n_stable_arr = np.zeros(N, dtype=np.int64)
    truncated_arr = np.zeros(N, dtype=bool)
    n_central    = central_pressure_grid(guidance_config).size
    M_arr        = np.full((N, n_central), np.nan)
    R_arr        = np.full((N, n_central), np.nan)
    L_arr        = np.full((N, n_central), np.nan)
    clamp_counts = {}

    # pQCD arrays (allocated regardless but only exported if pqcd_on)
    log_L_pqcd_arr = np.full(N, -np.inf, dtype=np.float64)
    pqcd_pass_arr  = np.zeros(N, dtype=bool)
    pqcd_n_T_arr   = np.full(N, np.nan,  dtype=np.float64)
    pqcd_mu_T_arr  = np.full(N, np.nan,  dtype=np.float64)

    # Detect whether either NICER path was active in the run -- used
    # below to decide whether to expose the split keys in the output.
    have_summary_pulsars = any(
        p.get("mode", "summary_gaussian") == "summary_gaussian"
        for p in (guidance_config.get("nicer_pulsars", []) or []))
    have_kde_pulsars     = any(
        p.get("mode", "summary_gaussian") == "kde"
        for p in (guidance_config.get("nicer_pulsars", []) or []))

    n_errors = 0
    first_err = None
    for n, (lL, info, err) in enumerate(results):
        if err is not None:
            n_errors += 1
            if first_err is None:
                first_err = err
            if verbose and n_errors <= 3:
                print(f"    sample {n}: SciPy TOV failed ({err}), weight=0")
            continue
        if np.isfinite(lL):
            log_L_exact[n] = lL
        log_L_nicer[n]         = info["log_L_nicer"]
        log_L_nicer_summary[n] = info.get("log_L_nicer_summary", 0.0)
        log_L_nicer_kde[n]     = info.get("log_L_nicer_kde",     0.0)
        log_L_gw[n]     = info["log_L_gw"]
        log_L_mmax[n]   = info["log_L_mmax"]
        M_max_arr[n]    = info["M_max_pred"]
        n_stable_arr[n] = info["n_stable"]
        truncated_arr[n] = info.get("truncated", False)
        M_arr[n] = info["M"]
        R_arr[n] = info["R"]
        L_arr[n] = info["Lambda"]
        for pname, hit in info["clamp_hits"].items():
            clamp_counts[pname] = clamp_counts.get(pname, 0) + int(hit)
        if pqcd_on:
            # info carries these only when pQCD is active
            log_L_pqcd_arr[n] = info.get("log_L_pqcd", -np.inf)
            pqcd_pass_arr[n]  = info.get("pqcd_passed", False)
            pqcd_n_T_arr[n]   = info.get("pqcd_n_T",  np.nan)
            pqcd_mu_T_arr[n]  = info.get("pqcd_mu_T", np.nan)

    if n_errors == N:
        raise RuntimeError(
            f"every one of the {N} samples raised inside the TOV/pQCD "
            f"evaluation, so M, R and Lambda are empty and no likelihood "
            f"was formed.  First error: {first_err}")

    # Guard on the physics: when every TOV solve returns all-NaN M, each
    # term still returns its finite floor, the weights come out uniform
    # and ESS == N -- a failed run that reads as a flawless one.
    n_nobranch = int((n_stable_arr < 2).sum())
    if n_nobranch == N:
        raise RuntimeError(
            f"no sample out of {N} has a stable branch with >= 2 points: "
            f"the TOV stage produced nothing, so there is no likelihood to "
            f"form.  {n_errors} sample(s) raised"
            + (f"; first error: {first_err}" if first_err is not None else "")
            + ".  Check that P[-1] from the thermodynamic integration "
              "exceeds P_c_min, and that every optional dependency of the "
              "pQCD path is installed.")

    # Truncated samples carry a lower bound on M_max, not M_max: reject
    # them from every term.  The unmasked joint is kept so the weight
    # cost of the policy can be measured after the fact.
    log_L_exact_premask = log_L_exact.copy()
    premask_terms = {
        "log_L_nicer": log_L_nicer.copy(),
        "log_L_gw":    log_L_gw.copy(),
        "log_L_mmax":  log_L_mmax.copy(),
    }
    if pqcd_on:
        premask_terms["log_L_pqcd"] = log_L_pqcd_arr.copy()
    n_trunc = int(truncated_arr.sum())
    trunc_weight_frac = 0.0
    if n_trunc:
        if np.any(np.isfinite(log_L_exact_premask)):
            _w_pre, _ = _ess_from_logw(log_L_exact_premask, N,
                                       context="pre-truncation joint")
            trunc_weight_frac = float(_w_pre[truncated_arr].sum())
        log_L_exact[truncated_arr] = -np.inf
        log_L_nicer[truncated_arr] = -np.inf
        log_L_nicer_summary[truncated_arr] = -np.inf
        log_L_nicer_kde[truncated_arr]     = -np.inf
        log_L_gw[truncated_arr]    = -np.inf
        log_L_mmax[truncated_arr]  = -np.inf
        if pqcd_on:
            log_L_pqcd_arr[truncated_arr] = -np.inf

    if verbose:
        print(f"  [reweighting] stable-branch census: "
              f"{n_trunc}/{N} truncated ({100.0 * n_trunc / N:.2f}%), "
              f"{n_nobranch}/{N} with no stable branch "
              f"({100.0 * n_nobranch / N:.2f}%), "
              f"{n_errors}/{N} raised ({100.0 * n_errors / N:.2f}%)")
        if n_trunc:
            print(f"  [reweighting] truncated samples rejected; they would "
                  f"have carried {100.0 * trunc_weight_frac:.3f}% of the "
                  f"posterior weight.  The cut is set by where the density "
                  f"grid stops, not by physics -- report both numbers in "
                  f"Methods.")
            if trunc_weight_frac > 0.01:
                print("  [reweighting] NOTE: > 1% of the posterior weight "
                      "was discarded by the truncation policy.  The "
                      "posterior is then partly a statement about the "
                      "density ceiling; consider extending truncated "
                      "samples with c_s^2 = 1 above the grid top, which "
                      "bounds M_max from above instead of discarding them.")

    if not np.any(np.isfinite(log_L_exact)):
        raise RuntimeError(
            f"no sample has a finite log-likelihood out of {N} "
            f"({n_errors} raised, {n_trunc} truncated, {n_nobranch} with no "
            f"stable branch); nothing to reweight.")

    # Joint + per-term weights + ESS
    weights,    ESS_total = _ess_from_logw(log_L_exact, N, context="joint")
    _,          ESS_nicer = _ess_from_logw(log_L_nicer, N, context="NICER")
    _,          ESS_gw    = _ess_from_logw(log_L_gw,    N, context="GW")
    _,          ESS_mmax  = _ess_from_logw(log_L_mmax,  N, context="M_max")

    out = {
        "log_L_exact":  log_L_exact,
        "log_L_nicer":  log_L_nicer,
        "log_L_gw":     log_L_gw,
        "log_L_mmax":   log_L_mmax,
        "weights":      weights,
        "log_weights":  log_L_exact,
        "ESS":          float(ESS_total),
        "ESS_nicer":    float(ESS_nicer),
        "ESS_gw":       float(ESS_gw),
        "ESS_mmax":     float(ESS_mmax),
        "N":            int(N),
        "M_max_pred":   M_max_arr,
        "n_stable":     n_stable_arr,
        "truncated":    truncated_arr,
        "log_L_exact_premask":       log_L_exact_premask,
        "premask_terms":             premask_terms,
        "truncated_weight_fraction": float(trunc_weight_frac),
        "n_no_stable_branch":        int(n_nobranch),
        "n_errors":                  int(n_errors),
        "M":            M_arr,
        "R":            R_arr,
        "Lambda":       L_arr,
        "clamp_counts": clamp_counts,
        "likelihood_version": LIKELIHOOD_VERSION,
    }

    # Expose the NICER summary/kde split only when at least one such
    # pulsar was active in the run -- keeps the output dict tidy when
    # the user is on pure summary mode (the pre-existing default).
    if have_summary_pulsars and have_kde_pulsars:
        _, ESS_nicer_summary = _ess_from_logw(log_L_nicer_summary, N,
                                              context="NICER summary")
        _, ESS_nicer_kde     = _ess_from_logw(log_L_nicer_kde,     N,
                                              context="NICER KDE")
        out["log_L_nicer_summary"] = log_L_nicer_summary
        out["log_L_nicer_kde"]     = log_L_nicer_kde
        out["ESS_nicer_summary"]   = float(ESS_nicer_summary)
        out["ESS_nicer_kde"]       = float(ESS_nicer_kde)
    elif have_kde_pulsars and not have_summary_pulsars:
        # All-KDE run: log_L_nicer_kde is identical to log_L_nicer,
        # but emit it under the kde key as well for downstream code
        # that wants to detect "this was a KDE run".
        _, ESS_nicer_kde = _ess_from_logw(log_L_nicer_kde, N,
                                          context="NICER KDE")
        out["log_L_nicer_kde"] = log_L_nicer_kde
        out["ESS_nicer_kde"]   = float(ESS_nicer_kde)

    # Add pQCD diagnostics only when pQCD is active.
    if pqcd_on:
        _, ESS_pqcd = _ess_from_logw(log_L_pqcd_arr, N)
        out["log_L_pqcd"]  = log_L_pqcd_arr
        out["pqcd_passed"] = pqcd_pass_arr
        out["pqcd_n_T"]    = pqcd_n_T_arr
        out["pqcd_mu_T"]   = pqcd_mu_T_arr
        out["ESS_pqcd"]    = float(ESS_pqcd)

    if verbose:
        n_finite = int(np.isfinite(log_L_exact).sum())
        print(f"  [reweighting] done.  {n_finite}/{N} finite log-L.")
        print(f"  [reweighting] ESS total = {ESS_total:.1f} / {N}  "
              f"({100.0 * ESS_total / N:.1f}%)")
        print(f"  [reweighting] per-term ESS (each data term alone):")
        print(f"                ESS_nicer = {ESS_nicer:.1f} / {N}  "
              f"({100.0 * ESS_nicer / N:.1f}%)")
        if "ESS_nicer_summary" in out:
            print(f"                  -- summary-Gaussian = "
                  f"{out['ESS_nicer_summary']:.1f}")
        if "ESS_nicer_kde" in out:
            print(f"                  -- KDE              = "
                  f"{out['ESS_nicer_kde']:.1f}")
        print(f"                ESS_gw    = {ESS_gw:.1f} / {N}  "
              f"({100.0 * ESS_gw / N:.1f}%)")
        print(f"                ESS_mmax  = {ESS_mmax:.1f} / {N}  "
              f"({100.0 * ESS_mmax / N:.1f}%)")
        if pqcd_on:
            n_pass = int(pqcd_pass_arr.sum())
            print(f"                ESS_pqcd  = {out['ESS_pqcd']:.1f} / {N}  "
                  f"({100.0 * out['ESS_pqcd'] / N:.1f}%)")
            print(f"  [reweighting] pQCD: {n_pass}/{N} samples got "
                  f"finite log-L ({100.0 * n_pass / N:.1f}%)")
        if clamp_counts:
            total_clamp = sum(clamp_counts.values())
            if total_clamp > 0:
                print(f"  [reweighting] NICER clamp hits (samples whose stable "
                      f"branch did not reach the pulsar):")
                for pname, cnt in clamp_counts.items():
                    pct = 100.0 * cnt / N
                    marker = ("   <-- effectively uninformative"
                              if pct > 50 else "")
                    print(f"                {pname}: {cnt}/{N}  "
                          f"({pct:.1f}%){marker}")

    return out


def weighted_quantile(values, weights, q):
    values  = np.asarray(values,  dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    mask = np.isfinite(values) & (weights > 0)
    if not mask.any():
        return float('nan')
    v = values[mask]
    w = weights[mask]
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, v))


def weighted_summary(samples_phys, weights):
    """
    Weighted point-wise mean, std, and 16/84 quantiles.

    Returns (mean, std, q16, q84) as a 4-tuple.
    """
    samples_phys = np.asarray(samples_phys, dtype=np.float64)
    weights      = np.asarray(weights,      dtype=np.float64)
    N, L = samples_phys.shape
    w = weights / weights.sum()
    mean_curve = (samples_phys * w[:, None]).sum(axis=0)
    # weighted std
    var_curve = (w[:, None] * (samples_phys - mean_curve[None, :]) ** 2
                 ).sum(axis=0)
    std_curve = np.sqrt(var_curve)
    q16 = np.array([weighted_quantile(samples_phys[:, j], weights, 0.16)
                    for j in range(L)])
    q84 = np.array([weighted_quantile(samples_phys[:, j], weights, 0.84)
                    for j in range(L)])
    return mean_curve, std_curve, q16, q84