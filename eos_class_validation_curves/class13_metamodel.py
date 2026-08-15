"""
Class 13 (validation family C): nuclear empirical meta-model
(metamodel core + causal high-density extension).

Core construction
The energy per nucleon of homogeneous nucleonic matter is a Taylor
expansion about the saturation density of symmetric matter,
parameterized by the nuclear empirical parameters (NEPs), quadratic in
the isospin asymmetry delta = 1 - 2 x_p (x_p = proton fraction):

    x        = (n - n_sat) / (3 n_sat)
    e_sat(n) = E_sat + K_sat x^2/2 + Q_sat x^3/6  + Z_sat x^4/24
    e_sym(n) = E_sym + L_sym x + K_sym x^2/2 + Q_sym x^3/6 + Z_sym x^4/24
    e(n, d)  = e_sat(n) + d^2 e_sym(n)

(the linear isoscalar term vanishes because n_sat is the saturation
minimum).  At order N = 4 this form reproduces binding energy, pressure
and sound speed of a large set of realistic EOS to percent accuracy up
to ~4 n_sat (Margueron et al. 2018, paper I).

beta-equilibrium with electrons and muons
-----------------------------------------
    mu_n - mu_p = -de/dx_p = 4 d e_sym(n) = mu_e ,   mu_mu = mu_e ,
    charge neutrality:  n x_p = n_e + n_mu ,
    n_l(mu) = (mu^2 - m_l^2)^{3/2} / (3 pi^2 (hbar c)^3)   for mu > m_l ,

solved for x_p(n) on the whole core grid at once by a vectorized
bisection: g(x_p) = n x_p - n_e - n_mu is strictly increasing in x_p
whenever e_sym > 0, with g(0) <= 0 and g(0.5) = n/2 > 0, so the root is
unique and bisection is guaranteed to converge (60 iterations,
|error| < 0.5 * 2^-60).  If e_sym(n) <= 0 the energy minimum over the
asymmetry sits at delta = 1, and the same bisection automatically
returns x_p = 0 (pure neutron matter, no leptons) -- the consistent
continuation, not a rejection.  Then

    eps(n) = n [ m_N + e(n, d(n)) ] + eps_e + eps_mu ,

with relativistic Fermi-gas lepton energy densities (vectorized form of
eos_common.fermi_energy_integral), and

    c_s^2 = dP/deps ,   P = n (deps/dn) - eps ,

via the shared finite-difference routine cs2_from_eps_array on the core
part of NB_GRID -- the same route as training class 6 (quarkyonic).

Matching construction (why the core stops at n_m)
-------------------------------------------------
The order-4 Taylor form is validated up to ~4 n_sat (paper I) and is
NOT designed for our full grid reaching 8 n0 ~ 8.3 n_sat: continued
naively, the K_sat x^2 growth alone drives c_s^2 through 1, and the
weakly constrained Q, Z terms destabilize eps(n).  Measured on this
grid, the bare Taylor form is acausal/unstable for ~99.8% of draws even
near the Margueron centres.  We therefore follow the standard usage of
the metamodel in neutron-star inference -- metamodel below a matching
density, causal sound-speed extension above (e.g. Koehn et al. 2024 and
Reed et al. 2024 employ it below 2 n_sat; hybrid frameworks anchor a
GP/polytrope high-density extension on a metamodel low-density part):

    n_m = t * n_sat ,  t ~ U(2, 4)       (matching density),
    n <= n_m : metamodel c_s^2 as above (validity checks on the core),
    n >  n_m : causal continuation, drawn among
               (i)   constant   c_s^2(n) = c_s^2(n_m),
               (ii)  linear in n from c_s^2(n_m) to c_end ~ U(0, 1),
               (iii) linear in n from c_s^2(n_m) to c_end ~ U(0.20,0.45)
                     (approach to the near-conformal band).

All three continuations are convex interpolations of values in [0, 1],
hence causal and stable by construction; the core is filtered by the
shared acceptance checks exactly as training classes 4 and 6.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.abspath(os.path.join(_HERE, os.pardir,
                                          "eos_training_curves"))
if os.path.isdir(_TRAIN_DIR) and _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from enum import Enum, auto
import numpy as np

from eos_common import (
    n0, mN, hc, hc3, NB_GRID, NB_OVER_N0_GRID, N_GRID,
    cs2_from_eps_array, cs2_acceptance_check,
)
from sampling_utils import uniform, beta_scaled, normal_truncated

CLASS_NAME = "Class 13: Nuclear Empirical Meta-Model (Margueron)"
FILE_PREFIX = "class13_metamodel"

M_E = 0.5110       # MeV
M_MU = 105.6584    # MeV

TMATCH_MIN, TMATCH_MAX = 2.0, 4.0

NSAT_MIN, NSAT_MAX = 0.15, 0.17          # fm^-3
ESAT_MIN, ESAT_MAX = -17.5, -14.5        # MeV
KSAT_MIN, KSAT_MAX = 190.0, 300.0
QSAT_MIN, QSAT_MAX = -1200.0, 1000.0
ZSAT_MIN, ZSAT_MAX = -4000.0, 5000.0
ESYM_MIN, ESYM_MAX = 27.0, 37.0
LSYM_MIN, LSYM_MAX = 20.0, 80.0
KSYM_MIN, KSYM_MAX = -400.0, 300.0
QSYM_MIN, QSYM_MAX = -2000.0, 5000.0
ZSYM_MIN, ZSYM_MAX = -5000.0, 5000.0

_MARG_MU = dict(nsat=0.155, Esat=-15.8, Ksat=230.0, Qsat=300.0,
                Zsat=-500.0, Esym=32.0, Lsym=60.0, Ksym=-100.0,
                Qsym=0.0, Zsym=-500.0)
_MARG_SG = dict(nsat=0.005, Esat=0.3, Ksat=20.0, Qsat=400.0,
                Zsat=1000.0, Esym=2.0, Lsym=15.0, Ksym=100.0,
                Qsym=400.0, Zsym=1000.0)

_BOUNDS = dict(nsat=(NSAT_MIN, NSAT_MAX), Esat=(ESAT_MIN, ESAT_MAX),
               Ksat=(KSAT_MIN, KSAT_MAX), Qsat=(QSAT_MIN, QSAT_MAX),
               Zsat=(ZSAT_MIN, ZSAT_MAX), Esym=(ESYM_MIN, ESYM_MAX),
               Lsym=(LSYM_MIN, LSYM_MAX), Ksym=(KSYM_MIN, KSYM_MAX),
               Qsym=(QSYM_MIN, QSYM_MAX), Zsym=(ZSYM_MIN, ZSYM_MAX))

_KEYS = ("nsat", "Esat", "Ksat", "Qsat", "Zsat",
         "Esym", "Lsym", "Ksym", "Qsym", "Zsym")


class Rejection(Enum):
    EPS_NONPOSITIVE = auto()
    P_NONPOSITIVE = auto()
    DE_NONPOSITIVE = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()
    UNSTABLE = auto()
    LOW_VARIATION = auto()

def _e_sat(x, p):
    return (p["Esat"] + 0.5 * p["Ksat"] * x**2
            + p["Qsat"] * x**3 / 6.0 + p["Zsat"] * x**4 / 24.0)


def _e_sym(x, p):
    return (p["Esym"] + p["Lsym"] * x + 0.5 * p["Ksym"] * x**2
            + p["Qsym"] * x**3 / 6.0 + p["Zsym"] * x**4 / 24.0)


def _lep_density_vec(mu, m):
    """n_l(mu) = (mu^2 - m^2)^{3/2} / (3 pi^2 (hbar c)^3)  [fm^-3];
    vectorized, 0 below threshold mu <= m (and for mu < 0)."""
    t = np.clip(mu * mu - m * m, 0.0, None)
    t = np.where(mu > m, t, 0.0)
    return t ** 1.5 / (3.0 * np.pi**2 * hc3)


def _fermi_energy_vec(kF, m):
    """Vectorized fermi_energy_integral of eos_common (identical
    formula); kF = 0 gives exactly 0 because ln((0+m)/m) = 0."""
    E = np.sqrt(kF * kF + m * m)
    return 0.125 * (kF * E * (2.0 * kF * kF + m * m)
                    - m**4 * np.log((kF + E) / m))


def _lep_eps_vec(n_l, m):
    """Relativistic Fermi-gas energy density of one lepton species."""
    kF = (3.0 * np.pi**2 * np.clip(n_l, 0.0, None)) ** (1.0 / 3.0) * hc
    return _fermi_energy_vec(kF, m) / (np.pi**2 * hc3)


def _solve_xp_vec(nB_arr, esym_arr, n_iter=60):
    """beta-equilibrium proton fraction on the core grid at once.

    Vectorized bisection on g(x_p) = n x_p - n_e(mu) - n_mu(mu) with
    mu = 4 (1 - 2 x_p) e_sym.  For e_sym > 0, g is strictly increasing
    with g(0) <= 0 < g(0.5); for e_sym <= 0 the leptons vanish and the
    iteration converges to x_p = 0 (pure neutron matter)."""
    lo = np.zeros_like(nB_arr)
    hi = np.full_like(nB_arr, 0.5)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        mu = 4.0 * (1.0 - 2.0 * mid) * esym_arr
        g = (nB_arr * mid
             - _lep_density_vec(mu, M_E) - _lep_density_vec(mu, M_MU))
        pos = g > 0.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def _solve_xp(nB, esym_val):
    """Scalar convenience wrapper (used by the smoke test)."""
    return float(_solve_xp_vec(np.array([nB]), np.array([esym_val]))[0])


def _core_cs2(p, i_m):
    """Metamodel c_s^2 on the core grid NB_GRID[: i_m + 1].
    Returns (cs2_core, None) or (None, Rejection.*)."""
    nb = NB_GRID[: i_m + 1]
    x = (nb - p["nsat"]) / (3.0 * p["nsat"])
    esat = _e_sat(x, p)
    esym = _e_sym(x, p)
    if (np.any(~np.isfinite(esat)) or np.any(~np.isfinite(esym))):
        return None, Rejection.NUMERICAL

    xp = _solve_xp_vec(nb, esym)
    delta = 1.0 - 2.0 * xp
    mu_e = 4.0 * delta * esym
    n_e = _lep_density_vec(mu_e, M_E)
    n_mu = _lep_density_vec(mu_e, M_MU)
    eps = (nb * (mN + esat + delta * delta * esym)
           + _lep_eps_vec(n_e, M_E) + _lep_eps_vec(n_mu, M_MU))

    if np.any(~np.isfinite(eps)):
        return None, Rejection.NUMERICAL
    if np.any(eps <= 0.0):
        return None, Rejection.EPS_NONPOSITIVE

    cs2, P_arr, tag = cs2_from_eps_array(eps, nb)
    if cs2 is None:
        return None, Rejection.DE_NONPOSITIVE
    if np.any(~np.isfinite(cs2)) or np.any(~np.isfinite(P_arr)):
        return None, Rejection.NUMERICAL
    if np.any(P_arr <= 0.0):
        return None, Rejection.P_NONPOSITIVE
    if np.any(cs2 > 1.0):
        return None, Rejection.ACAUSAL
    if np.any(cs2 < 0.0):
        return None, Rejection.UNSTABLE

    return cs2, None


def _extend_cs2(cs2_core, i_m, rng):
    """Causal continuation above the matching point: constant, linear to
    a free endpoint in [0,1], or linear toward the near-conformal band.
    Convex interpolation of values in [0,1] -> stays in [0,1]."""
    n_tail = N_GRID - (i_m + 1)
    c_m = float(np.clip(cs2_core[-1], 0.0, 1.0))
    kind = int(rng.integers(3))
    if kind == 0:
        tail = np.full(n_tail, c_m)
    elif kind == 1:
        c_end = uniform(0.0, 1.0, rng=rng)
        tail = np.linspace(c_m, c_end, n_tail + 1)[1:]
    else:
        c_end = uniform(0.20, 0.45, rng=rng)
        tail = np.linspace(c_m, c_end, n_tail + 1)[1:]
    return np.concatenate([cs2_core, tail])

def _flat(rng, **over):
    p = {}
    for k in _KEYS:
        lo, hi = over.get(k, _BOUNDS[k])
        p[k] = uniform(lo, hi, rng=rng)
    return p


def _s_reference_flat(rng=None):
    """Table-I flat prior box of arXiv:2112.09595, unmodified."""
    return _flat(rng)


def _s_margueron_gaussian(rng=None):
    """Margueron 2018 empirical centres; truncated at the flat box."""
    return {k: normal_truncated(_MARG_MU[k], _MARG_SG[k],
                                _BOUNDS[k][0], _BOUNDS[k][1], rng=rng)
            for k in _KEYS}


def _s_low_order(rng=None):
    """Minimal (order-2) metamodel: Q = Z = 0 in both channels."""
    p = _flat(rng)
    p["Qsat"] = p["Zsat"] = p["Qsym"] = p["Zsym"] = 0.0
    return p


def _s_stiff_isoscalar(rng=None):
    return _flat(rng, Ksat=(250.0, KSAT_MAX), Qsat=(0.0, QSAT_MAX))


def _s_soft_isoscalar(rng=None):
    return _flat(rng, Ksat=(KSAT_MIN, 240.0), Qsat=(QSAT_MIN, 0.0))


def _s_stiff_isovector(rng=None):
    return _flat(rng, Lsym=(55.0, LSYM_MAX), Ksym=(0.0, KSYM_MAX))


def _s_soft_isovector(rng=None):
    return _flat(rng, Lsym=(LSYM_MIN, 45.0), Ksym=(KSYM_MIN, -100.0))


def _s_extended_lsym(rng=None):
    """L_sym widened to [10, 120] MeV (beyond the Table-I box),
    bracketing PREX-era determinations (project: wider priors)."""
    return _flat(rng, Lsym=(10.0, 120.0))


def _s_wild_high_order(rng=None):
    """Push the weakly constrained Q, Z parameters to the box edges."""
    def _edge(lo, hi):
        third = (hi - lo) / 3.0
        return ((lo, lo + third) if uniform(0, 1, rng=rng) < 0.5
                else (hi - third, hi))
    return _flat(rng, Qsat=_edge(QSAT_MIN, QSAT_MAX),
                 Zsat=_edge(ZSAT_MIN, ZSAT_MAX),
                 Qsym=_edge(QSYM_MIN, QSYM_MAX),
                 Zsym=_edge(ZSYM_MIN, ZSYM_MAX))


def _s_beta_centered(rng=None):
    return {k: beta_scaled(2.0, 2.0, _BOUNDS[k][0], _BOUNDS[k][1],
                           rng=rng)
            for k in _KEYS}


STRATEGIES = [
    _s_reference_flat, _s_margueron_gaussian, _s_low_order,
    _s_stiff_isoscalar, _s_soft_isoscalar,
    _s_stiff_isovector, _s_soft_isovector,
    _s_extended_lsym, _s_wild_high_order,
    _s_beta_centered,
]


def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    params = strategy(rng=rng)
    t_match = uniform(TMATCH_MIN, TMATCH_MAX, rng=rng)
    xm = t_match * params["nsat"] / n0
    i_m = int(np.searchsorted(NB_OVER_N0_GRID, xm))
    i_m = int(np.clip(i_m, 12, N_GRID - 12))   # >= 12 pts core and tail

    cs2_core, reason = _core_cs2(params, i_m)
    if cs2_core is None:
        return None, strategy.__name__, reason

    cs2 = _extend_cs2(cs2_core, i_m, rng)

    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]

    return cs2, strategy.__name__, None
if __name__ == "__main__":
    import time
    from collections import Counter
    from sampling_utils import set_seed

    rng = set_seed(123)
    N_TEST = 300
    ok, rej, strat = [], Counter(), Counter()
    tries = 0
    t0 = time.time()
    while len(ok) < N_TEST and tries < 400 * N_TEST:
        tries += 1
        c, s, r = generate_one_sample(rng=rng)
        if c is None:
            rej[r] += 1
        else:
            ok.append(c)
            strat[s] += 1
    dt = time.time() - t0

    arr = np.array(ok)
    assert arr.shape == (N_TEST, N_GRID), f"bad shape {arr.shape}"
    assert np.all(np.isfinite(arr))
    assert arr.min() >= 0.0 and arr.max() <= 1.0, "causality violated!"
    assert arr.std(axis=1).min() >= 0.01, "low-variation curve slipped in"

    p0 = {k: _MARG_MU[k] for k in _KEYS}
    x0 = (n0 - p0["nsat"]) / (3.0 * p0["nsat"])
    es0 = float(_e_sym(np.array([x0]), p0)[0])
    xp0 = _solve_xp(n0, es0)
    print(f"{CLASS_NAME}")
    print(f"  accepted {len(ok)}/{tries} ({len(ok)/tries:.1%})  "
          f"[{1e3*dt/tries:.2f} ms/attempt]")
    print(f"  c_s^2 in [{arr.min():.3e}, {arr.max():.6f}]  "
          f"min per-curve std = {arr.std(axis=1).min():.4f}")
    print(f"  sanity (Margueron centres): x_p(n0) = {xp0:.4f}  "
          f"mu_e = {4*(1-2*xp0)*es0:.1f} MeV")
    for s, c in strat.most_common():
        print(f"    {s:32s}: {c}")
    for r, c in rej.most_common():
        print(f"    REJ {r.name:24s}: {c}")
    print("  smoke test PASSED")
