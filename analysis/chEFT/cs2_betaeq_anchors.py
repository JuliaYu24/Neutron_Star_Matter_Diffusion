"""
Beta-equilibrium speed-of-sound conditioning anchors and their covariance.

Computes c_s^2(n) for charge-neutral, beta-equilibrated npe-mu matter on a set
of anchor densities, together with the anchor-anchor covariance, for use as the
conditioning input of the denoising-diffusion EOS reconstruction.

Inputs and statistical model
----------------------------
The chiral-EFT MBPT energies per particle for pure neutron matter (PNM, E_N)
and symmetric nuclear matter (SNM, E_S) are supplied by the BUQEYE
nuclear-matter-convergence package [Drischler, Melendez, Furnstahl, Phillips,
PRC 102, 054315 (2020); Drischler, Furnstahl, Melendez, Phillips, PRL 125,
202702 (2020)]. Interpolation + EFT-truncation uncertainties are Gaussian
processes; the PNM-SNM channels are correlated through the coregionalized
symmetry-energy kernel of the package's `SymmetryEnergyContainer`, with the
per-cutoff settings of the official `analysis/derivatives-bands.ipynb`:

    Lambda = 500 MeV : rho = None  (geometric coregionalization)
    Lambda = 450 MeV : rho = 0.95, ls_sym = (ls_n + ls_s)/2   (official choice)

The PNM-SNM cross-covariance at deriv-0 is recovered from the three covariances
the package exposes (PNM, SNM, difference S2 = E_N - E_S) by the polarization
identity

    Cov(E_N, E_S) = 1/2 [ Cov(E_N, E_N) + Cov(E_S, E_S) - Cov(S2, S2) ],

which is exact for the truncation part (verified to machine precision against a
direct construction of the coregionalization cross-kernel) and defines the
unique symmetric-cross joint Gaussian whose three marginals match the BUQEYE
containers exactly. The assembled 2N x 2N covariance over [E_N(grid),
E_S(grid)] is PSD up to a numerical rim of O(1e-4 MeV^2) from the tiny
interpolation-posterior blocks; the sampler clips negative eigenvalues.

Correlated energy curves are drawn, density derivatives are taken numerically
on the fine grid (validated per-sample against the GP's analytic derivative
channel at the 0.1% level of the fluctuation scale), and the beta-equilibrium
layer is applied. A diagonal (independent) draw sharing the same random numbers
is produced alongside as a sensitivity check on the PNM-SNM correlation.

Beta-equilibrium layer (standard construction, cf. Drischler, Han, Lattimer,
Prakash, Reddy, Zhao, PRC 103, 045808 (2021))
----------------------------------------------------------------------------
Parabolic (quadratic) isospin expansion with S2 = E_N - E_S:

    e_N(n, x) = (1-x) m_n + x m_p + E_S(n) + S2(n) (1-2x)^2     [per baryon]

Neutrino-free equilibrium mu_n - mu_p = mu_e and mu_mu = mu_e give

    mu_e = (m_n - m_p) + 4 S2 (1 - 2x),

solved together with charge neutrality x n = n_e(mu_e) + n_mu(mu_e) by a
vectorized bisection (the residual is monotone in mu_e). The baryon chemical
potential along the catalyzed trajectory is

    mu_B = d eps / dn = e_N + n (d e_N/dn)|_x + mu_e x,

where the composition derivatives cancel by the equilibrium (envelope)
condition -- the mu_e x term carries the entire lepton + rearrangement
contribution. The squared sound speed of the barotropic beta-stable EOS is

    c_s^2 = dP/deps = (n / mu_B) d mu_B / dn        (Gibbs-Duhem at T = 0).

Both identities, and the equivalence with the explicit route
eps = n e_N + eps_lep, P = n mu_B - eps, c_s^2 = P'(n)/eps'(n), are checked
numerically on the mean curves at every run (`validate=True`).

Self-contained layout -- no repository clone or environment variables needed.
This module expects to live in a directory that also contains
    nuclear_matter/                     (the BUQEYE package folder)
    all_matter_data_high_density.csv    (the input data table)
and finds both automatically, regardless of the current working directory.
An explicit data path can still be passed (csv_path=... / --csv / MATTER_CSV).
Runs in any environment where the original notebooks run (gsum, gptools);
the compatibility shim below covers Python 3.11+ / NumPy 2.x / modern SciPy.

Usage
-----
    python cs2_betaeq_anchors.py [--outdir products]

Running the file with no arguments processes BOTH cutoffs, 500 and 450 MeV,
and writes the full set of products for each; every filename carries its
Lambda tag, so the two never collide.  Restrict with --Lambda 500 (or
--Lambda 450) for a single cutoff.  When both run, a comparison table of the
anchor means, sigmas and covariance is printed at the end.  The validation
against the published PNM sample set applies to Lambda = 500 only and is
skipped automatically for 450.
"""

# -----------------------------------------------------------------------------
# Legacy-stack compatibility shim.
# gptools / gsum predate modern SciPy, NumPy 2.x and Python 3.11+ and call names
# that have since been removed (scipy.integer, np.NaN, inspect.getargspec, Py2
# builtins, ...). The block below restores them. Every assignment is guarded, so
# it is a no-op where the names already exist. It must run before
# `import nuclear_matter`, which triggers the gptools import chain.
# -----------------------------------------------------------------------------
import builtins as _bi, warnings as _warn, inspect as _inspect  # noqa: E401
import io as _io, contextlib as _ctx
import numpy as np
import scipy as _scipy
_warn.filterwarnings("ignore")

# Python 2 builtins still referenced by the legacy code
for _n, _v in dict(long=int, unicode=str, basestring=str, xrange=range).items():
    if not hasattr(_bi, _n):
        setattr(_bi, _n, _v)

# NumPy >= 2.0 removed aliases
for _n, _v in dict(NaN=np.nan, NAN=np.nan, Inf=np.inf, infty=np.inf, NINF=-np.inf,
                   float=float, int=int, bool=bool, object=object, complex=complex,
                   str=str, unicode_=str).items():
    if not hasattr(np, _n):
        try:
            setattr(np, _n, _v)
        except Exception:
            pass
for _n, _v in dict(alltrue=np.all, sometrue=np.any, product=np.prod).items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)

# scipy.<name> -> numpy.<name> fallback (scipy.integer, scipy.floating, ...)
try:
    _orig_scipy_getattr = _scipy.__getattr__
except AttributeError:
    _orig_scipy_getattr = None
def _scipy_getattr(name):
    if _orig_scipy_getattr is not None:
        try:
            return _orig_scipy_getattr(name)
        except AttributeError:
            pass
    if hasattr(np, name):
        return getattr(np, name)
    raise AttributeError(f"module 'scipy' has no attribute {name!r}")
_scipy.__getattr__ = _scipy_getattr

# inspect.getargspec was removed in Python 3.11
if not hasattr(_inspect, "getargspec"):
    from collections import namedtuple as _nt
    _ArgSpec = _nt("ArgSpec", ["args", "varargs", "keywords", "defaults"])
    def _getargspec(func):
        s = _inspect.getfullargspec(func)
        return _ArgSpec(s.args, s.varargs, s.varkw, s.defaults)
    _inspect.getargspec = _getargspec
# --------------------------- end compatibility shim --------------------------

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
try:
    import nuclear_matter  # noqa: F401  (pip-installed or already on sys.path)
except ImportError:
    _sys.path.insert(0, _HERE)          # use the package folder next to this file

from nuclear_matter import fermi_momentum, InputData
from nuclear_matter.derivatives import ObservableContainer, SymmetryEnergyContainer

# =============================================================================
# Constants and GP hyperparameters
# =============================================================================
HBARC = 197.3269804          # MeV fm
M_N, M_P = 939.565, 938.272  # MeV
M_E, M_MU = 0.51099895, 105.6583755
DMN = M_N - M_P              # 1.293 MeV
N_SAT = 0.16                 # fm^-3, reference saturation density

# Marginal variance (std = cbar) and correlation length (ls, in fm^-1 of the
# respective Fermi momentum) from the order-by-order fits in the official
# BUQEYE analysis/derivatives-bands.ipynb, together with that notebook's
# per-cutoff settings for the symmetry-energy (difference) container.
HYPERPARAMS = {
    500: dict(std_n=1.00, ls_n=0.973,
              std_s=2.95, ls_s=0.484,
              rho=None,                         # geometric coregionalization
              ls_n_sym=0.973, ls_s_sym=0.484),
    450: dict(std_n=0.8684060649936118, ls_n=0.7631421388401067,
              std_s=2.6146499024837073, ls_s=0.46603268529311087,
              rho=0.95,                         # fixed cross-correlation
              ls_n_sym=(0.7631421388401067 + 0.46603268529311087) / 2,
              ls_s_sym=None),
}
ORDERS = np.array([0, 2, 3, 4])   # LO, NLO, N2LO, N3LO
BREAKDOWN = 600                    # MeV, EFT breakdown scale Lambda_b

_CSV_NAME = "all_matter_data_high_density.csv"


def find_data_csv(csv_path=None):
    """Resolve the data table: explicit path > $MATTER_CSV > next to this
    module > current working directory."""
    candidates = [
        csv_path or "",
        _os.environ.get("MATTER_CSV", ""),
        _os.path.join(_HERE, _CSV_NAME),
        _CSV_NAME,
    ]
    found = next((p for p in candidates if p and _os.path.isfile(p)), None)
    if found is None:
        raise FileNotFoundError(
            f"could not find {_CSV_NAME!r}; place it next to "
            f"cs2_betaeq_anchors.py (looked in {_HERE}) or pass csv_path=...")
    return found


def _pred0(container, order):
    """Deriv-0 posterior mean; falls back to the private cache on old package copies."""
    try:
        return container.get_pred(order, 0)
    except AttributeError:
        return container._y_interp_vecs[order][0]


# =============================================================================
# Beta-equilibrium layer
# =============================================================================
def _n_lepton_vec(mu):
    """Total lepton number density n_e + n_mu (fm^-3) for chemical potential mu (MeV)."""
    out = np.zeros_like(mu, dtype=float)
    for m in (M_E, M_MU):
        pF2 = mu * mu - m * m
        pF = np.sqrt(np.clip(pF2, 0.0, None))
        out += np.where(pF2 > 0.0, pF**3 / (3 * np.pi**2 * HBARC**3), 0.0)
    return out


def _eps_lepton_vec(mu):
    """Total lepton energy density (MeV/fm^3) for chemical potential mu (MeV)."""
    eps = np.zeros_like(mu, dtype=float)
    for m in (M_E, M_MU):
        pF2 = mu * mu - m * m
        pF = np.sqrt(np.clip(pF2, 0.0, None))
        ash = np.where(pF2 > 0.0, np.arcsinh(pF / np.maximum(m, 1e-30)), 0.0)
        E = np.sqrt(pF * pF + m * m)
        term = (1.0 / (np.pi**2 * HBARC**3)) * (1.0 / 8.0) * (
            pF * (2 * pF * pF + m * m) * E - m**4 * ash)
        eps += np.where(pF2 > 0.0, term, 0.0)
    return eps


def solve_proton_fraction(n_grid, S2, n_iter=90):
    """Beta-equilibrium proton fraction x and electron chemical potential mu_e.

    Vectorized bisection on mu_e; the residual x*n - n_lepton is monotone
    (decreasing) in mu_e, with a sign change guaranteed on the bracket
    [m_e, DMN + 4 S2]. n_grid: (N,) fm^-3; S2: (M,N) or (N,) MeV.
    Returns x, mu_e with S2's shape.
    """
    n_b = np.broadcast_to(n_grid, S2.shape)
    lo = np.full_like(S2, M_E * 1.0001)
    hi = DMN + 4.0 * S2
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        x = 0.5 * (1.0 - (mid - DMN) / (4.0 * S2))
        res = x * n_b - _n_lepton_vec(mid)
        hi = np.where(res < 0.0, mid, hi)
        lo = np.where(res >= 0.0, mid, lo)
    mu_e = 0.5 * (lo + hi)
    x = np.clip(0.5 * (1.0 - (mu_e - DMN) / (4.0 * S2)), 0.0, 0.5)
    return x, mu_e


def betaeq_thermo(n_grid, E_snm, dE_snm_dn, S2, dS2_dn, full=False):
    """Thermodynamics of charge-neutral, beta-equilibrated npe-mu matter.

    Parabolic symmetry-energy approximation. Energies in MeV per baryon
    (WITHOUT rest mass); derivatives with respect to density (MeV fm^3).
    Shapes: n_grid (N,); the rest (M, N) for M samples, or (N,).

    Returns a dict with cs2, x, mu_e, mu_B and -- if ``full`` -- also the
    total energy density eps (MeV/fm^3) and pressure P = n mu_B - eps.
    """
    x, mu_e = solve_proton_fraction(n_grid, S2)
    f = 1.0 - 2.0 * x
    e_N = (1 - x) * M_N + x * M_P + E_snm + S2 * f * f   # per-baryon energy, incl. rest mass
    dEm_dn = dE_snm_dn + f * f * dS2_dn                  # d e_N/dn at fixed x
    mu_B = e_N + n_grid * dEm_dn + mu_e * x              # = d eps/dn (dx/dn cancels at beta-eq)
    cs2 = (n_grid / mu_B) * np.gradient(mu_B, n_grid, axis=-1)
    out = dict(cs2=cs2, x=x, mu_e=mu_e, mu_B=mu_B)
    if full:
        eps = n_grid * e_N + _eps_lepton_vec(mu_e)       # total energy density
        out["eps"] = eps
        out["P"] = n_grid * mu_B - eps                   # Euler relation
    return out


def cs2_beta_equilibrium(n_grid, E_snm, dE_snm_dn, S2, dS2_dn):
    """Back-compatible wrapper: returns (cs2, x, mu_e)."""
    t = betaeq_thermo(n_grid, E_snm, dE_snm_dn, S2, dS2_dn)
    return t["cs2"], t["x"], t["mu_e"]


def validate_betaeq(n_grid, E_snm, dE_snm_dn, S2, dS2_dn, interior=2):
    """Numerical consistency checks of the beta-eq layer on given curves.

    (1) envelope: d(eps)/dn == mu_B; (2) Gibbs-Duhem: dP/dn == n d(mu_B)/dn;
    (3) cs2 route equivalence: (n/mu_B) mu_B'  ==  P'(n)/eps'(n).
    All residuals are finite-difference-limited; the small remainder is
    concentrated at the muon-onset kink. Returns a dict of max deviations
    over interior grid points.
    """
    t = betaeq_thermo(n_grid, E_snm, dE_snm_dn, S2, dS2_dn, full=True)
    sl = slice(interior, -interior)
    deps = np.gradient(t["eps"], n_grid, axis=-1)
    dP = np.gradient(t["P"], n_grid, axis=-1)
    dmu = np.gradient(t["mu_B"], n_grid, axis=-1)
    cs2_direct = dP / deps
    return dict(
        envelope_rel=float(np.max(np.abs(deps - t["mu_B"])[..., sl] / t["mu_B"][..., sl])),
        gibbs_duhem_MeV=float(np.max(np.abs(dP - n_grid * dmu)[..., sl])),
        cs2_routes_rel=float(np.max(
            np.abs(t["cs2"] - cs2_direct)[..., sl] / np.abs(cs2_direct)[..., sl])),
        muon_onset_fm3=float(n_grid[np.argmax(np.atleast_2d(t["mu_e"])[0] > M_MU)]),
    )


# =============================================================================
# Joint (correlated) anchor-covariance extraction
# =============================================================================
_TRAIN_GRID_CACHE = [None]


def training_grid():
    """The 200-point output grid the network is trained on, in units of n0.

    Read (never written) from eos_common.NB_OVER_N0_GRID when that module is
    importable, so there is a single source of truth; otherwise rebuilt from
    the same three numbers.  Nothing on the training side is modified.
    """
    if _TRAIN_GRID_CACHE[0] is None:
        try:
            from eos_common import NB_OVER_N0_GRID as _g
            g = np.asarray(_g, dtype=float)
        except Exception:
            g = np.linspace(0.5, 8.0, 200)      # == eos_common.NB_OVER_N0_GRID
        _TRAIN_GRID_CACHE[0] = g
    return _TRAIN_GRID_CACHE[0]


def _snap_anchor_densities(targets, grid=None):
    """Move requested anchor densities [n0] onto the nearest training-grid points.

    A conditioning value only means something if it sits on a component of the
    200-point output vector, so anchors quoted at round numbers have to be
    moved.  Returns (densities [n0], component indices).
    """
    g = training_grid() if grid is None else np.asarray(grid, dtype=float)
    t = np.atleast_1d(np.asarray(targets, dtype=float))
    half = 0.5 * float(np.min(np.diff(g)))
    if t.min() < g[0] - half or t.max() > g[-1] + half:
        raise ValueError(f"anchors {t} leave the training grid "
                         f"[{g[0]:g}, {g[-1]:g}] n0")
    j = np.clip(np.searchsorted(g, t), 1, g.size - 1)
    idx = np.where(t - g[j - 1] <= g[j] - t, j - 1, j).astype(int)
    if np.unique(idx).size != idx.size:
        raise ValueError(f"anchors {t} collapse onto the same grid point "
                         f"(step {2 * half:g} n0); space them further apart")
    return g[idx], idx


def anchor_grid_map(targets=(0.5, 0.75, 1.0, 1.25, 1.5), grid=None):
    """Table of the requested -> on-grid anchor mapping (diagnostic)."""
    used, idx = _snap_anchor_densities(targets, grid=grid)
    req = np.atleast_1d(np.asarray(targets, dtype=float))
    head = (f"{'requested':>10} {'index':>6} {'on grid':>11} "
            f"{'shift':>9} {'rel':>8} {'nB [fm^-3]':>11}")
    lines = [head, "-" * len(head)]
    for r, i, v in zip(req, idx, used):
        lines.append(f"{r:10.4f} {i:6d} {v:11.6f} {v - r:+9.5f} "
                     f"{(v - r) / r:+7.2%} {v * N_SAT:11.6f}")
    return "\n".join(lines)


def aligned_grid(lo, hi, pad=2, refine=1, grid=None):
    """The training grid, restricted to [lo, hi] and padded at both ends.

    `pad` extra points beyond each end keep the anchors off the boundary,
    where np.gradient degrades to a one-sided difference.  This matters:
    anchor 0 sits exactly on the first training point, and with pad=0 its
    c_s^2 comes out several percent wrong.  `refine` subdivides the step;
    refine=1 is already converged, so raising it only costs time.

    Do NOT build this by unioning the two grids.  They are incommensurate
    (step ratio 240/199), so a union puts points 2.5e-5 fm^-3 apart -- a
    199:1 spacing ratio that np.gradient divides by.
    """
    g = training_grid() if grid is None else np.asarray(grid, dtype=float)
    g = g * N_SAT
    g = g[(g >= lo - 1e-12) & (g <= hi + 1e-12)]
    if g.size < 2:
        raise ValueError(f"no training-grid points inside [{lo:g}, {hi:g}] fm^-3")
    if refine > 1:
        g = np.concatenate([np.linspace(g[i], g[i + 1], refine + 1)[:-1]
                            for i in range(g.size - 1)] + [g[-1:]])
    st = g[1] - g[0]
    d = np.concatenate([g[0] - st * np.arange(pad, 0, -1), g,
                        g[-1] + st * np.arange(1, pad + 1)])
    return d[(d >= lo - 1e-12) & (d <= hi + 1e-12)]


def _interp_at(x, Y, xt, atol=1e-12):
    """Linear interpolation of Y (..., len(x)) at targets xt along the last axis.

    Degenerates to exact indexing where a target coincides with a node, so a
    NaN in a neighbouring column cannot leak through the zero weight.
    """
    x = np.asarray(x, dtype=float)
    xt = np.atleast_1d(np.asarray(xt, dtype=float))
    if xt.min() < x[0] or xt.max() > x[-1]:
        raise ValueError(f"targets outside [{x[0]:g}, {x[-1]:g}]")
    j = np.clip(np.searchsorted(x, xt), 1, x.size - 1)
    w = (xt - x[j - 1]) / (x[j] - x[j - 1])
    out = (1.0 - w) * Y[..., j - 1] + w * Y[..., j]
    hi = np.abs(x[j] - xt) <= atol
    if np.any(hi):
        out[..., hi] = Y[..., j[hi]]
    lo = np.abs(x[j - 1] - xt) <= atol
    if np.any(lo):
        out[..., lo] = Y[..., j[lo] - 1]
    return out


def _symm_psd_sqrt(M):
    """Symmetric PSD square root with eigenvalue clipping (stable sampler)."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T


def build_joint_gp(csv_path=None, Lambda=500, dn_grid=0.005, order=4,
                   align_to_training_grid=True):
    """Build the three BUQEYE containers and assemble the joint PNM-SNM model.

    With align_to_training_grid (default) the in-range points of
    eos_common.NB_OVER_N0_GRID are merged into the evaluation grid, so the GP
    posterior is reported exactly where the network expects it and no
    interpolation is needed at the anchors.  dn_grid is then unused.  Set
    False to reproduce the pre-alignment behaviour.

    Follows the official analysis/derivatives-bands.ipynb conventions per
    cutoff (hyperparameters, err_y_d = sqrt(err_n^2 + err_s^2), include_3bf =
    False, breakdown = 600 MeV, orders LO/NLO/N2LO/N3LO). Returns a dict with
    the fine grid, means, the three covariance blocks, the polarization cross
    block C, and diagnostics (pointwise rho, min eigenvalue of the joint).
    """
    if Lambda not in HYPERPARAMS:
        raise ValueError("Lambda must be 450 or 500")
    hp = HYPERPARAMS[Lambda]

    data = InputData(find_data_csv(csv_path), Lambda)
    density = data.density
    # NB: on the BUQEYE high-density table this reproduces the official
    # 59-point fine grid 0.05 ... 0.34 fm^-3 (inclusive, via float rounding).
    density_all = np.arange(density[0], density[-1], dn_grid)
    if align_to_training_grid:
        density_all = aligned_grid(density[0], density[-1])
        if np.min(np.abs(density_all - N_SAT)) > 1e-9:
            density_all = np.sort(np.append(density_all, N_SAT))
    kf_n_all = fermi_momentum(density_all, 2)
    kf_s_all = fermi_momentum(density_all, 4)
    ref_n = 16 / fermi_momentum(N_SAT, 2)**2
    ref_s = 16 / fermi_momentum(N_SAT, 4)**2

    min_unc, unc_fac = 0.02, 0.001
    def errfn(yv):
        e = np.abs(yv[:, -1]) * unc_fac
        e[e < min_unc] = min_unc
        return e
    err_n = errfn(data.y_n_2_plus_3bf)
    err_s = errfn(data.y_s_2_plus_3bf)
    err_d = np.sqrt(err_n**2 + err_s**2)      # official difference-channel choice

    obs_n = ObservableContainer(
        density=density, kf=data.kf_n, y=data.y_n_2_plus_3bf, orders=ORDERS,
        density_interp=density_all, kf_interp=kf_n_all, std=hp["std_n"], ls=hp["ls_n"],
        ref=ref_n, breakdown=BREAKDOWN, err_y=err_n,
        include_3bf=False, derivs=[0], verbose=False)
    obs_s = ObservableContainer(
        density=density, kf=data.kf_s, y=data.y_s_2_plus_3bf, orders=ORDERS,
        density_interp=density_all, kf_interp=kf_s_all, std=hp["std_s"], ls=hp["ls_s"],
        ref=ref_s, breakdown=BREAKDOWN, err_y=err_s,
        include_3bf=False, derivs=[0], verbose=False)
    with _ctx.redirect_stdout(_io.StringIO()):   # suppress a stray print in the package
        obs_d = SymmetryEnergyContainer(
            density=density, y=data.y_d_2_plus_3bf, orders=ORDERS,
            density_interp=density_all,
            std_n=hp["std_n"], ls_n=hp["ls_n_sym"],
            std_s=hp["std_s"], ls_s=hp["ls_s_sym"],
            ref_n=ref_n, ref_s=ref_s, breakdown=BREAKDOWN, err_y=err_d,
            include_3bf=False, derivs=[0], verbose=False, rho=hp["rho"])

    # deriv-0 covariance blocks (interpolation + truncation) and the
    # polarization cross block
    S_nn = obs_n.get_cov(order, 0, 0)             # Cov(E_N, E_N)
    S_ss = obs_s.get_cov(order, 0, 0)             # Cov(E_S, E_S)
    S_dd = obs_d.get_cov(order, 0, 0)             # Cov(S2, S2) = S_nn + S_ss - 2C
    C = 0.5 * (S_nn + S_ss - S_dd)                # Cov(E_N, E_S)
    mu_n = _pred0(obs_n, order)
    mu_s = _pred0(obs_s, order)

    joint = np.block([[S_nn, C], [C, S_ss]])
    min_eig = float(np.linalg.eigvalsh(0.5 * (joint + joint.T)).min())
    rho_point = np.diag(C) / np.sqrt(np.diag(S_nn) * np.diag(S_ss))

    return dict(density_all=density_all, mu_n=mu_n, mu_s=mu_s,
                S_nn=S_nn, S_ss=S_ss, S_dd=S_dd, C=C,
                rho_point=rho_point, min_eig_joint=min_eig, Lambda=Lambda)


def extract_betaeq_anchor_cov(
    csv_path=None,
    Lambda=500,
    anchors_n_over_n0=(0.5, 0.75, 1.0, 1.25, 1.5),
    snap_to_training_grid=True,
    num_samp=20000,
    seed=12345,
    dn_grid=0.005,
    n_ref=0.08,
    validate=True,
    keep_samples=True,
    gp=None,
):
    """Joint (correlated) and diagonal (independent) beta-eq c_s^2 anchor stats.

    Randomness is shared between the two draws, so the only difference is the
    PNM-SNM cross-covariance. Pass a precomputed ``gp = build_joint_gp(...)``
    to skip the container build (e.g. when varying seed/num_samp).

    With ``snap_to_training_grid`` (default) the requested anchor densities
    are moved to the nearest point of the 200-point training grid
    (eos_common.NB_OVER_N0_GRID = linspace(0.5, 8.0, 200)), since a
    conditioning value only means something if it sits on a component of the
    output vector.  (0.5, 0.75, 1.0, 1.25, 1.5) becomes (0.5, 0.763819,
    0.989950, 1.253769, 1.517588), i.e. grid indices (0, 7, 13, 20, 27), the
    largest move being +1.8% in density.  The values used and their component
    indices come back as ``anchors_n_over_n0`` and ``grid_index``, the
    originals as ``anchors_requested``.  Pass False for the old behaviour.

    Returns a dict with the anchor densities, the joint-draw mean and
    covariance (the conditioning inputs), the independent-draw covariance
    (diagnostic), full-grid summary curves for plotting, physics diagnostics
    (x_p, mu_e, Sv, L, pressure at n0), and the validation residuals. With
    ``keep_samples`` (default), the joint-draw E_SNM samples (num_samp, N) and
    the density grid are kept under ``samples`` for save_sample_products.
    ``n_ref`` (snapped to the grid) sets the low-density reference point: the
    thermodynamic integration constants (mu_B, eps, P) there are evaluated on
    the same joint samples and returned under ``refpoint``, including their
    joint mean/covariance with the cs2 anchors (see save_reference_point).
    """
    if gp is None:
        gp = build_joint_gp(csv_path, Lambda=Lambda, dn_grid=dn_grid)
    density_all = gp["density_all"]
    mu_n, mu_s = gp["mu_n"], gp["mu_s"]
    S_nn, S_ss, C = gp["S_nn"], gp["S_ss"], gp["C"]
    N = len(density_all)
    rng = np.random.default_rng(seed)

    # --- joint vs independent draw (shared random numbers) --------------------
    Zero = np.zeros((N, N))
    L_joint = _symm_psd_sqrt(np.block([[S_nn, C], [C, S_ss]]))
    L_indep = _symm_psd_sqrt(np.block([[S_nn, Zero], [Zero, S_ss]]))
    g = rng.standard_normal((2 * N, num_samp))
    mu_stack = np.concatenate([mu_n, mu_s])[:, None]

    def _run(L, full=False):
        s = L @ g
        E_N = (mu_stack[:N] + s[:N]).T            # (num_samp, N)
        E_S = (mu_stack[N:] + s[N:]).T
        S2 = E_N - E_S
        S2 = np.where(S2 <= 1e-6, 1e-6, S2)       # guard rare non-physical draws
        dE_S = np.gradient(E_S, density_all, axis=-1)
        dS2 = np.gradient(S2, density_all, axis=-1)
        t = betaeq_thermo(density_all, E_S, dE_S, S2, dS2, full=full)
        return t, E_S, S2, dS2

    t_joint, E_S_joint, S2_joint, dS2_joint = _run(L_joint, full=True)  # full: eps/P for the reference point
    t_indep, _, _, _ = _run(L_indep)

    # --- anchor densities on the training grid --------------------------------
    # The model outputs c_s^2 on eos_common.NB_OVER_N0_GRID = linspace(0.5,
    # 8.0, 200), so a conditioning anchor has to sit exactly on one of those
    # points; otherwise it is attached to a component whose density differs by
    # up to half a grid step.  Snap the requested densities to the nearest
    # grid point and carry the component index downstream.
    req = np.atleast_1d(np.asarray(anchors_n_over_n0, dtype=float))
    if snap_to_training_grid:
        anchors_used, gidx = _snap_anchor_densities(req)
    else:
        anchors_used, gidx = req, np.full(req.shape, -1, dtype=int)
    nB = anchors_used * N_SAT
    if nB.min() < density_all[0] or nB.max() > density_all[-1]:
        raise ValueError(
            f"anchor densities {np.round(nB, 5)} fm^-3 leave the chiral-EFT "
            f"grid [{density_all[0]:g}, {density_all[-1]:g}] fm^-3")

    # The chiral grid (dn_grid = 0.005 fm^-3) is not a refinement of the
    # training grid (step 0.00603 fm^-3), so a nearest-index lookup would give
    # back up to 0.0025 fm^-3 of the snap.  Evaluate the per-draw curves at nB
    # by linear interpolation instead -- a linear map of the samples, so the
    # anchor covariance stays exact.
    _exact = np.min(np.abs(density_all[:, None] - nB[None, :]), axis=0) <= 1e-12
    diagnostics_exact = int(_exact.sum())

    def _at_anchors(Y):
        return _interp_at(density_all, Y, nB)

    def _stats(cs2):
        s = _at_anchors(cs2)
        good = ~np.any(np.isnan(s), axis=1)
        s = s[good]
        return s.mean(0), np.cov(s, rowvar=False), int(good.sum()), good

    mean_j, cov_j, ng_j, good_j = _stats(t_joint["cs2"])
    mean_i, cov_i, ng_i, _ = _stats(t_indep["cs2"])

    # --- physics diagnostics on the mean curves -------------------------------
    S2_mean = mu_n - mu_s
    dEs_mean = np.gradient(mu_s, density_all)
    dS2_mean = np.gradient(S2_mean, density_all)
    tm = betaeq_thermo(density_all, mu_s, dEs_mean, S2_mean, dS2_mean, full=True)
    j0 = int(np.argmin(np.abs(density_all - N_SAT)))
    diagnostics = dict(
        Sv=float(S2_mean[j0]),
        L=float(3 * N_SAT * dS2_mean[j0]),
        x_n0=float(tm["x"][j0]),
        mu_e_n0=float(tm["mu_e"][j0]),
        P_beta_n0=float(tm["P"][j0]),
        eps_n0=float(tm["eps"][j0]),
        E_snm_n0=float(mu_s[j0]),
    )
    # per-draw slope L(draw) = 3 n0 dS2/dn|n0 from the JOINT draw: the
    # covariance-consistent L distribution (PNM-SNM truncation errors
    # correlated within each draw -> their partial cancellation is built in).
    L_samples = 3.0 * N_SAT * dS2_joint[:, j0]
    _Lg = L_samples[np.isfinite(L_samples)]
    diagnostics['L_samples_median'] = float(np.median(_Lg))
    diagnostics['L_samples_q16_84'] = (float(np.percentile(_Lg, 16)),
                                       float(np.percentile(_Lg, 84)))
    diagnostics['L_samples_std'] = float(np.std(_Lg, ddof=1))
    validation = (validate_betaeq(density_all, mu_s, dEs_mean, S2_mean, dS2_mean)
                  if validate else None)

    # --- full-grid summary curves (joint draw) for plotting --------------------
    curves = dict(
        density=density_all,
        cs2_mean=np.nanmean(t_joint["cs2"], axis=0),
        cs2_std=np.nanstd(t_joint["cs2"], axis=0),
        cs2_std_indep=np.nanstd(t_indep["cs2"], axis=0),
        x_mean=t_joint["x"].mean(0), x_std=t_joint["x"].std(0),
        mu_e_mean=t_joint["mu_e"].mean(0), mu_e_std=t_joint["mu_e"].std(0),
        rho_point=gp["rho_point"],
    )

    samples = (dict(E_snm=E_S_joint, S2=S2_joint, L_samples=L_samples,
                    density=density_all)
               if keep_samples else None)

    # --- low-density reference point (same joint samples as the anchors) ------
    jref = int(np.argmin(np.abs(density_all - n_ref)))
    mu_r = t_joint["mu_B"][good_j, jref]
    eps_r = t_joint["eps"][good_j, jref]
    P_r = t_joint["P"][good_j, jref]
    stack = np.column_stack([mu_r, eps_r, _at_anchors(t_joint["cs2"][good_j])])
    refpoint = dict(
        n_ref=float(density_all[jref]),
        quantities=("mu_B", "eps", "P"),                 # MeV, MeV/fm^3, MeV/fm^3
        mean=np.array([mu_r.mean(), eps_r.mean(), P_r.mean()]),
        sigma=np.array([mu_r.std(ddof=1), eps_r.std(ddof=1), P_r.std(ddof=1)]),
        joint_labels=["mu_B_ref", "eps_ref"]
                     + [f"cs2@{a:.4f}n0" for a in anchors_used],
        joint_mean=stack.mean(0),
        joint_cov=np.cov(stack, rowvar=False),
    )

    return dict(
        nB=nB, anchors_n_over_n0=anchors_used,
        anchors_requested=req, grid_index=gidx,
        anchors_on_chiral_grid=_exact,
        cs2_mean=mean_j, cov_joint=cov_j, cov_indep=cov_i,
        sigma_joint=np.sqrt(np.diag(cov_j)), sigma_indep=np.sqrt(np.diag(cov_i)),
        cs2_mean_indep=mean_i,
        rho_point_at_anchors=_at_anchors(gp["rho_point"]),
        clean_joint=ng_j, clean_indep=ng_i, Lambda=gp["Lambda"],
        min_eig_joint=gp["min_eig_joint"],
        diagnostics=diagnostics, validation=validation, curves=curves,
        samples=samples, refpoint=refpoint,
    )


# =============================================================================
# Reporting helpers
# =============================================================================
def print_report(res):
    """Human-readable summary of an extract_betaeq_anchor_cov result."""
    d, v = res["diagnostics"], res["validation"]
    print(f"Lambda = {res['Lambda']} MeV, N3LO, npe-mu beta equilibrium")
    print(f"clean samples: joint={res['clean_joint']}, indep={res['clean_indep']}; "
          f"min eig(joint 2Nx2N) = {res['min_eig_joint']:.2e} MeV^2")
    print(f"implied pointwise PNM-SNM rho at anchors: "
          f"{np.round(res['rho_point_at_anchors'], 4)}")
    _ex = res.get("anchors_on_chiral_grid")
    if _ex is not None:
        print(f"anchors evaluated exactly on the chi-EFT grid: "
              f"{int(np.sum(_ex))}/{len(_ex)}"
              + ("" if np.all(_ex) else "  (rest interpolated)"))
    rp = res.get("refpoint")
    if rp is not None:
        m, s = rp["mean"], rp["sigma"]
        print(f"reference point at n_ref = {rp['n_ref']:.3f} fm^-3: "
              f"mu_B = {m[0]:.2f} +/- {s[0]:.2f} MeV, "
              f"eps = {m[1]:.3f} +/- {s[1]:.3f}, "
              f"P = {m[2]:.4f} +/- {s[2]:.4f} MeV/fm^3")
    print()
    print(f"mean-curve diagnostics at n0 = {N_SAT} fm^-3:")
    print(f"  Sv = {d['Sv']:.2f} MeV, L = {d['L']:.1f} MeV, E_SNM = {d['E_snm_n0']:.2f} MeV")
    if "L_samples_median" in d:
        _lo, _hi = d["L_samples_q16_84"]
        print(f"  L (covariance-consistent, per-draw): "
              f"{d['L_samples_median']:.1f} +{_hi - d['L_samples_median']:.1f}"
              f"/-{d['L_samples_median'] - _lo:.1f} MeV  (68%), std {d['L_samples_std']:.1f}")
    print(f"  x_p = {d['x_n0']:.4f}, mu_e = {d['mu_e_n0']:.2f} MeV, "
          f"P_beta = {d['P_beta_n0']:.3f} MeV/fm^3")
    if v is not None:
        print(f"consistency: |deps/dn - mu_B|/mu_B <= {v['envelope_rel']:.1e}; "
              f"|dP/dn - n dmu_B/dn| <= {v['gibbs_duhem_MeV']:.1e} MeV;")
        print(f"  cs2 (mu_B route) vs dP/deps <= {v['cs2_routes_rel']:.1e} rel "
              f"(residual peaks at the muon onset, {v['muon_onset_fm3']:.3f} fm^-3)\n")
    _gi = res.get("grid_index")
    _ex = res.get("anchors_on_chiral_grid")
    print(f"{'comp':>5} {'n/n0':>10} {'nB':>9} {'cs2 mean':>10} "
          f"{'sig indep':>11} {'sig joint':>11} {'ratio':>7} {'':>3}")
    print("-" * 72)
    for k, r in enumerate(res["anchors_n_over_n0"]):
        si, sj = res["sigma_indep"][k], res["sigma_joint"][k]
        c = "-" if _gi is None or int(_gi[k]) < 0 else str(int(_gi[k]))
        flag = "" if _ex is None else ("" if _ex[k] else " ~")
        print(f"{c:>5} {r:>10.6f} {r * N_SAT:>9.6f} {res['cs2_mean'][k]:>10.5f} "
              f"{si:>11.5f} {sj:>11.5f} {sj/si:>7.3f} {flag:>3}")
    if _ex is not None and not np.all(_ex):
        print("  ~ = interpolated (not on the chi-EFT grid)")
    corr = res["cov_joint"] / np.outer(res["sigma_joint"], res["sigma_joint"])
    print("\njoint anchor correlation matrix:")
    with np.printoptions(precision=3, suppress=True):
        print(corr)


def torch_block(res, precision=10):
    """Format the joint anchor mean and covariance as torch tensors to paste in."""
    f = lambda v: f"{v:.{precision}e}"  # noqa: E731
    # full precision: the anchors sit on training-grid points, so a rounded
    # value would no longer match NB_OVER_N0_GRID exactly
    labels = ", ".join(f"{r:.12g}" for r in res["anchors_n_over_n0"])
    gidx = res.get("grid_index")
    lines = []
    lines.append("# " + "=" * 68)
    lines.append(f"# {precision}-digit BETA-EQUILIBRIUM conditioning vectors "
                 f"(Lambda = {res['Lambda']} MeV, N3LO npe-mu, joint draw)")
    lines.append("# " + "=" * 68)
    if gidx is not None and np.all(np.asarray(gidx) >= 0):
        rq = res.get("anchors_requested")
        if rq is not None:
            lines.append("# requested " + ", ".join(f"{r:g}" for r in rq)
                         + " n0, snapped to eos_common.NB_OVER_N0_GRID")
        lines.append(f"idx_known = torch.tensor("
                     f"[{', '.join(str(int(i)) for i in gidx)}])"
                     f"   # components of the 200-point output vector")
    lines.append(f"nB_known  = torch.tensor([{labels}])")
    lines.append(f"cs2_known = torch.tensor([{', '.join(f(m) for m in res['cs2_mean'])}])")
    lines.append("cs2_sigma = None")
    lines.append("cs2_cov   = torch.tensor([")
    for row in res["cov_joint"]:
        lines.append(f"    [{', '.join(f(v) for v in row)}],")
    lines.append("])")
    return "\n".join(lines)


_torch_block = torch_block  # back-compatible alias


def save_products(res, outdir="."):
    """Save the conditioning arrays with the established file names."""
    import os
    L = res["Lambda"]
    paths = {}
    for tag, arr in (("cov", res["cov_joint"]), ("mean", res["cs2_mean"]),
                     ("nB", res["nB"]),
                     ("gridindex", res.get("grid_index"))):
        if arr is None:
            continue
        p = os.path.join(outdir, f"cs2_BETAEQ_Lambda-{L}_anchor_{tag}_5pt.npy")
        np.save(p, arr)
        paths[tag] = p
    return paths


def save_sample_products(res, outdir="."):
    """Regenerate the sample-level files consumed downstream (names and shapes
    exactly as produced by the original extraction notebook), now from the
    JOINT draw so they match the anchor covariance:

        EA_SNM_Lambda-{L}_samples_N3LO.npy    (num_samp, N)  E/A of SNM, no rest mass [MeV]
        S2_BETAEQ_Lambda-{L}_samples_N3LO.npy (num_samp, N)  symmetry energy S2=E_N-E_S [MeV]
        L_BETAEQ_Lambda-{L}_samples_N3LO.npy  (num_samp,)    per-draw slope L at n0 [MeV]
        nB_density_Lambda-{L}.npy             (N,)           density grid [fm^-3]

    S2_BETAEQ is the per-draw symmetry energy from the JOINT (correlated)
    draw, so its spread carries the covariance-consistent PNM-SNM truncation
    uncertainty; L_BETAEQ is the matching per-draw slope 3 n0 dS2/dn|n0.
    (xp_BETAEQ_* is still not produced; unused downstream.)
    Requires a result from extract_betaeq_anchor_cov(..., keep_samples=True).
    """
    s = res.get("samples")
    if s is None:
        raise ValueError("no samples stored -- rerun extract_betaeq_anchor_cov"
                         "(..., keep_samples=True)")
    L = res["Lambda"]
    items = ((f"EA_SNM_Lambda-{L}_samples_N3LO.npy", s["E_snm"]),
             (f"S2_BETAEQ_Lambda-{L}_samples_N3LO.npy", s["S2"]),
             (f"L_BETAEQ_Lambda-{L}_samples_N3LO.npy", s["L_samples"]),
             (f"nB_density_Lambda-{L}.npy", s["density"]))
    paths = {}
    for name, arr in items:
        p = _os.path.join(outdir, name)
        np.save(p, arr)
        paths[name] = p
    return paths


def save_reference_point(res, outdir="."):
    """Save the reference-point arrays with the established file names."""
    rp = res["refpoint"]
    L = res["Lambda"]
    paths = {}
    for tag, arr in (("mean",  rp["mean"]),                 # (3,) mu_B, eps, P
                     ("sigma", rp["sigma"]),                # (3,)
                     ("nB",    np.array([rp["n_ref"]]))):   # (1,) fm^-3
        p = _os.path.join(outdir, f"cs2_BETAEQ_Lambda-{L}_refpoint_{tag}.npy")
        np.save(p, arr)
        paths[tag] = p
    return paths


_PNM_SAMPLES_NAME = "pressure_cs2_samples.csv"


def find_pnm_samples(path=None):
    """Resolve the published BUQEYE PNM sample file (see
    check_pnm_against_published): explicit path > next to this module > cwd."""
    candidates = [path or "",
                  _os.path.join(_HERE, _PNM_SAMPLES_NAME),
                  _PNM_SAMPLES_NAME]
    found = next((p for p in candidates if p and _os.path.isfile(p)), None)
    if found is None:
        raise FileNotFoundError(
            f"could not find {_PNM_SAMPLES_NAME!r}; download it from "
            "https://raw.githubusercontent.com/buqeye/nuclear-matter-convergence/"
            "master/analysis/pressure_cs2_samples.csv and place it next to "
            "cs2_betaeq_anchors.py")
    return found


def check_pnm_against_published(samples_csv=None, csv_path=None, Lambda=500,
                                num_samp=20000, seed=12345, gp=None,
                                make_plot=True, outdir=".", xlim="anchors"):
    """End-to-end validation in PNM-only mode against the published samples.

    Draws E_PNM curves from this pipeline's GP (interpolation + truncation
    covariance), forms mu = E + m_n + n dE/dn and cs2 = (n/mu) dmu/dn with the
    same sampling and numerical-derivative route used for the beta-equilibrium
    anchors, and compares the per-density mean and standard deviation with the
    1000 published Monte Carlo samples released by the BUQEYE collaboration
    with their analysis code (analysis/pressure_cs2_samples.csv in
    github.com/buqeye/nuclear-matter-convergence, accompanying
    PRL 125, 202702 (2020) and PRC 102, 054315 (2020); N3LO,
    Lambda = 500 MeV, Lambda_b = 600 MeV, identical 59-point density grid).

    The published set exists for Lambda = 500 MeV only, and the achievable
    agreement is limited by its own Monte-Carlo precision:
    ~ (sigma/mu)/sqrt(1000) for the mean and 1/sqrt(2*999) ~ 2.2% for the
    width. Deviations at the grid edges are dominated by the one-sided finite
    differences there and are excluded from the deviation panel of the plot.

    The comparison is always COMPUTED on the full 0.05-0.34 fm^-3 grid: the
    anchors must lie in the interior, since np.gradient degrades to one-sided
    differences at the endpoints and cs2 involves two nested derivatives
    (truncating the grid at 0.5 n0 / 1.5 n0 biases the outermost anchors by up
    to 0.7 sigma and shrinks their width by ~27%). ``xlim`` only sets the
    plotted range: "anchors" (default) zooms to the anchor span, "full" shows
    the whole grid, or pass an explicit (lo, hi) tuple in fm^-3.

    Returns a dict with per-density statistics and deviations, the anchor-level
    summary, and (if make_plot) the matplotlib figure and saved file paths.
    """
    if Lambda != 500:
        raise ValueError("the published PNM sample set corresponds to Lambda = 500 MeV")
    import csv as _csvmod
    from collections import defaultdict

    spath = find_pnm_samples(samples_csv)

    # published samples first: they fix the grid this comparison has to run on
    by_n = defaultdict(list)
    with open(spath) as f:
        for row in _csvmod.DictReader(f):
            by_n[round(float(row["Density"]), 3)].append(float(row["cs2"]))
    n_pub = np.array(sorted(by_n))
    n_per = len(by_n[n_pub[0]])

    def _matches(gp_):
        if gp_ is None:
            return False
        d = gp_["density_all"]
        return (len(d) == len(n_pub)
                and np.max(np.abs(n_pub - np.round(d, 3))) <= 1e-9)

    if not _matches(gp):
        # The anchor pipeline runs on the training-aligned grid, but the
        # published samples live on the chiral 0.005 fm^-3 grid.  Rebuild on
        # that grid rather than comparing curves point-by-point across two
        # different grids.
        gp = build_joint_gp(csv_path, Lambda=Lambda,
                            align_to_training_grid=False)
        if not _matches(gp):
            raise ValueError(
                f"published sample grid ({len(n_pub)} points, "
                f"{n_pub[0]:g}..{n_pub[-1]:g} fm^-3) does not match the "
                f"unaligned pipeline grid ({len(gp['density_all'])} points, "
                f"{gp['density_all'][0]:g}..{gp['density_all'][-1]:g})")

    n = gp["density_all"]
    N = len(n)

    # this pipeline, PNM-only: same sampling + derivative route as the anchors
    Lc = _symm_psd_sqrt(gp["S_nn"])
    g = np.random.default_rng(seed).standard_normal((N, num_samp))
    E = (gp["mu_n"][:, None] + Lc @ g).T                    # (num_samp, N)
    mu = E + M_N + n * np.gradient(E, n, axis=-1)           # d eps/dn, eps = n(E + m_n)
    cs2 = (n / mu) * np.gradient(mu, n, axis=-1)
    m_pipe, s_pipe = cs2.mean(0), cs2.std(0, ddof=1)
    m_pub = np.array([np.mean(by_n[x]) for x in n_pub])
    s_pub = np.array([np.std(by_n[x], ddof=1) for x in n_pub])

    dev_m = m_pipe / m_pub - 1.0
    dev_s = s_pipe / s_pub - 1.0
    # report at the same densities the anchors actually use (training grid)
    _a_used, _ = _snap_anchor_densities((0.5, 0.75, 1.0, 1.25, 1.5))
    aidx = [int(np.argmin(np.abs(n - a * N_SAT))) for a in _a_used]
    noise_m = float(np.mean(s_pub / m_pub) / np.sqrt(n_per))
    noise_s = float(1.0 / np.sqrt(2.0 * (n_per - 1.0)))

    print(f"PNM-only validation vs published BUQEYE samples "
          f"({n_per} samples x {N} densities, Lambda=500, N3LO):")
    print(f"{'n':>7} {'mean(pipe)':>11} {'mean(publ)':>11} {'dev':>7} "
          f"{'sig(pipe)':>10} {'sig(publ)':>10} {'dev':>7}")
    for j in aidx:
        print(f"{n[j]:7.3f} {m_pipe[j]:11.5f} {m_pub[j]:11.5f} {dev_m[j]:7.2%} "
              f"{s_pipe[j]:10.5f} {s_pub[j]:10.5f} {dev_s[j]:7.2%}")
    print(f"anchors: max |mean dev| = {np.max(np.abs(np.array(dev_m)[aidx])):.2%}, "
          f"max |width dev| = {np.max(np.abs(np.array(dev_s)[aidx])):.2%}")
    print(f"MC precision of the published set: ~{noise_m:.2%} (mean), "
          f"~{noise_s:.2%} (width)")

    out = dict(density=n, mean_pipeline=m_pipe, sigma_pipeline=s_pipe,
               mean_published=m_pub, sigma_published=s_pub,
               dev_mean=dev_m, dev_sigma=dev_s, anchor_idx=aidx,
               n_published=n_per, mc_noise_mean=noise_m, mc_noise_sigma=noise_s,
               samples_csv=spath)

    if make_plot:
        import matplotlib.pyplot as plt
        if xlim == "anchors":
            lo, hi = n[aidx[0]] - 0.005, 2.0 * N_SAT
        elif xlim == "full":
            lo, hi = n[0], n[-1]
        else:
            lo, hi = xlim
        fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
        ax1.fill_between(n, m_pub - s_pub, m_pub + s_pub, color="C1", alpha=0.35,
                         label=fr"published $\pm1\sigma$ ({n_per} samples)")
        ax1.plot(n, m_pub, "C1-", lw=1.2)
        ax1.plot(n, m_pipe, "C0--", lw=1.6, label="this pipeline (mean)")
        ax1.plot(n, m_pipe - s_pipe, "C0:", lw=1.0)
        ax1.plot(n, m_pipe + s_pipe, "C0:", lw=1.0, label=r"this pipeline $\pm1\sigma$")
        ax1.plot(n[aidx], m_pipe[aidx], "ko", ms=4, label="anchor densities")
        ax1.set_xlabel(r"$n$ [fm$^{-3}$]")
        ax1.set_ylabel(r"$c_s^2$  (PNM)")
        ax1.set_xlim(lo, hi)
        sel = (n >= lo) & (n <= hi)
        ax1.set_ylim(0, 1.08 * max(m_pub[sel].max(), (m_pipe + s_pipe)[sel].max()))
        ax1.legend(frameon=False, fontsize=9, loc="upper left")
        axtop = ax1.secondary_xaxis("top", functions=(lambda v: v / N_SAT,
                                                      lambda v: v * N_SAT))
        axtop.set_xlabel(r"$n/n_0$")
        fig.tight_layout()
        tag = "" if xlim == "anchors" else "_full"
        paths = {}
        for ext in ("pdf", "png"):
            pth = _os.path.join(outdir, f"pnm_validation_Lambda-500{tag}.{ext}")
            fig.savefig(pth, dpi=200, bbox_inches="tight")
            paths[ext] = pth
        print("figure:", paths["pdf"])
        out["fig"] = fig
        out["figure_paths"] = paths

    return out


# =============================================================================
#  Command line: run one or both cutoffs in a single invocation
# =============================================================================
ALL_LAMBDAS = (500, 450)


def run_one(Lambda, args, csv):
    """Full product pipeline for a single cutoff.  Returns the result dict."""
    bar = "=" * 78
    print(f"\n{bar}\n  Lambda = {Lambda} MeV\n{bar}")

    gp = build_joint_gp(csv, Lambda=Lambda)
    res = extract_betaeq_anchor_cov(csv, Lambda=Lambda,
                                    num_samp=args.num_samp, seed=args.seed,
                                    n_ref=args.n_ref,
                                    keep_samples=not args.no_samples, gp=gp)
    print_report(res)

    paths = save_products(res, args.outdir)
    print("\nsaved:", *paths.values(), sep="\n  ")
    if not args.no_samples:
        spaths = save_sample_products(res, args.outdir)
        print("sample-level products (joint draw):", *spaths.values(), sep="\n  ")
    if not args.no_refpoint:
        rpaths = save_reference_point(res, args.outdir)
        print("reference point:", *rpaths.values(), sep="\n  ")

    if not args.no_pnm_check:
        if Lambda == 500:
            try:
                spath = find_pnm_samples(args.pnm_samples)
            except FileNotFoundError as e:
                print(f"\n[pnm validation skipped] {e}")
            else:
                print()
                try:
                    check_pnm_against_published(
                        samples_csv=spath, csv_path=csv, Lambda=Lambda, gp=gp,
                        num_samp=args.num_samp, make_plot=True,
                        outdir=args.outdir)
                except Exception as exc:
                    print(f"[pnm validation failed] "
                          f"{type(exc).__name__}: {exc}")
                    print("  anchor products above are unaffected.")
        else:
            print(f"\n[pnm validation] the published sample set exists for "
                  f"Lambda=500 only; skipped for Lambda={Lambda}.")

    print("\n" + torch_block(res))
    return res


def compare_cutoffs(results):
    """Side-by-side summary of the anchor products across cutoffs."""
    Ls = sorted(results, reverse=True)
    if len(Ls) < 2:
        return
    bar = "=" * 78
    print(f"\n{bar}\n  CUTOFF COMPARISON ({' vs '.join(f'{L} MeV' for L in Ls)})\n{bar}")

    ref = results[Ls[0]]
    same = all(np.array_equal(results[L]["grid_index"], ref["grid_index"])
               for L in Ls)
    print("anchor densities [n0]: "
          + ", ".join(f"{v:.6f}" for v in ref["anchors_n_over_n0"]))
    print("training-grid indices: "
          + ", ".join(str(int(i)) for i in ref["grid_index"])
          + ("   (identical across cutoffs)" if same else
             "   *** DIFFER ACROSS CUTOFFS -- investigate ***"))

    head = f"{'n/n0':>9}" + "".join(
        f"{'mean ' + str(L):>13}{'sigma ' + str(L):>13}" for L in Ls)
    print("\n" + head)
    print("-" * len(head))
    for k, u in enumerate(ref["anchors_n_over_n0"]):
        row = f"{u:9.4f}"
        for L in Ls:
            r = results[L]
            row += f"{r['cs2_mean'][k]:13.6f}{r['sigma_joint'][k]:13.6f}"
        print(row)

    if len(Ls) == 2:
        a, b = results[Ls[0]], results[Ls[1]]
        dm = b["cs2_mean"] - a["cs2_mean"]
        ds = b["sigma_joint"] / a["sigma_joint"] - 1.0
        print(f"\ncutoff sensitivity ({Ls[1]} vs {Ls[0]}):")
        print("  d(mean)          : "
              + ", ".join(f"{v:+.6f}" for v in dm))
        print("  d(mean) / sigma  : "
              + ", ".join(f"{v:+.3f}" for v in dm / a["sigma_joint"]))
        print("  sigma ratio - 1  : "
              + ", ".join(f"{v:+.2%}" for v in ds))
        Ca, Cb = a["cov_joint"], b["cov_joint"]
        print(f"  ||dC||_F/||C||_F : "
              f"{np.linalg.norm(Cb - Ca) / np.linalg.norm(Ca):.2%}")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        epilog="By default both cutoffs (500 and 450 MeV) are processed in "
               "one run; every output filename carries its Lambda tag, so "
               "they never collide.  Use --Lambda 500 for a single cutoff.")
    ap.add_argument("--csv", default=None,
                    help="path to the data table (default: found next to this script)")
    ap.add_argument("--Lambda", type=int, nargs="+", choices=ALL_LAMBDAS,
                    default=list(ALL_LAMBDAS), metavar="{450,500}",
                    help="cutoff(s) in MeV; default: both (500 450)")
    ap.add_argument("--num-samp", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n-ref", type=float, default=0.08,
                    help="reference-point density in fm^-3 (snapped to grid)")
    ap.add_argument("--no-samples", action="store_true",
                    help="skip the legacy sample-level .npy products")
    ap.add_argument("--no-refpoint", action="store_true",
                    help="skip the reference-point products")
    ap.add_argument("--pnm-samples", default=None,
                    help="path to the published pressure_cs2_samples.csv "
                         "(default: found next to this script)")
    ap.add_argument("--no-pnm-check", action="store_true",
                    help="skip the validation against the published PNM samples")
    ap.add_argument("--keep-going", action="store_true",
                    help="if one cutoff fails, carry on with the rest "
                         "(the exit status still reports the failure)")
    args = ap.parse_args(argv)

    lambdas = list(dict.fromkeys(args.Lambda))       # de-duplicate, keep order
    csv = find_data_csv(args.csv)
    _os.makedirs(args.outdir, exist_ok=True)
    print(f"using data: {csv}")
    print(f"cutoffs   : {', '.join(f'{L} MeV' for L in lambdas)}")
    print(f"outdir    : {_os.path.abspath(args.outdir)}")

    results, failed = {}, {}
    for L in lambdas:
        try:
            results[L] = run_one(L, args, csv)
        except Exception as exc:                      # noqa: BLE001
            failed[L] = exc
            print(f"\n*** Lambda = {L} MeV FAILED: "
                  f"{type(exc).__name__}: {exc}")
            if not args.keep_going:
                import traceback
                traceback.print_exc()
                break

    compare_cutoffs(results)

    print("\n" + "=" * 78)
    if results:
        print("completed: " + ", ".join(f"{L} MeV" for L in sorted(results, reverse=True)))
    if failed:
        print("FAILED   : " + ", ".join(f"{L} MeV ({type(e).__name__})"
                                        for L, e in failed.items()))
        return 1
    return 0


if __name__ == "__main__":
    _sys.exit(main())
