"""
Class 1: Piecewise Polytropes: P(rho) = K_i ρ^{gamma_i}

Uses a DIRECT rho -> nB mapping (rho = m_B x nB). This is exact for piecewise polytropes because P(rho), epsilon(rho) are known analytically, and rho is linear in nB.
"""

from enum import Enum, auto
import numpy as np
from eos_common import EPS_REF, P_REF, NB_REF, n0, NB_GRID, cs2_acceptance_check
from sampling_utils import (
    uniform, log_uniform, beta_scaled, normal_truncated,
    sorted_log_uniform, sorted_uniform,
    pick_count, alternating_values, one_peaked,
    uniform_excluding,
)


m_B = 931.494
RHO_REF = m_B * NB_REF

RHO_GRID = m_B * NB_GRID

CLASS_NAME = "Class 1: Piecewise Polytropes"
FILE_PREFIX = "class1_polytrope"


GAMMA_MIN, GAMMA_MAX = 0.5, 8.0
GAMMA_AVOID = 0.02
NSEG_OPTIONS = (3, 4, 5, 6, 7, 8, 9)
NB_BOUNDARY_MIN = 0.6 * n0
NB_BOUNDARY_MAX = 7.5 * n0


class Rejection(Enum):
    K_FAILED = auto()
    A_FAILED = auto()
    EPS_NONPOSITIVE = auto()
    EPS_PLUS_P_NONPOSITIVE = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()
    UNSTABLE = auto()
    LOW_VARIATION = auto()


def _fix_K(Gammas, rho_boundaries):
    """K_i from P continuity, anchored to P_REF at RHO_REF."""
    N_seg = len(Gammas)
    Ks = np.zeros(N_seg)
    Ks[0] = P_REF / (RHO_REF ** Gammas[0])
    if not (np.isfinite(Ks[0]) and Ks[0] > 0):
        return None
    for i in range(1, N_seg):
        Ks[i] = Ks[i - 1] * rho_boundaries[i - 1] ** (Gammas[i - 1] - Gammas[i])
        if not (np.isfinite(Ks[i]) and Ks[i] > 0):
            return None
    return Ks


def _eps_single(rho, K, G, a_const):
    """
    Energy density epsilon(rho) for one or more segments.
    All arguments may be scalars or arrays of the same shape.
    """
    rho, K, G, a_const = (np.asarray(x, dtype=float) for x in (rho, K, G, a_const))
    if rho.ndim == 0:
        if abs(float(G) - 1.0) < 1e-12:
            return float((1.0 + a_const) * rho + K * rho * np.log(rho))
        return float((1.0 + a_const) * rho + K / (G - 1.0) * rho ** G)
    result = np.empty_like(rho)
    close = np.abs(G - 1.0) < 1e-12
    if np.any(close):
        m = close
        result[m] = (1.0 + a_const[m]) * rho[m] + K[m] * rho[m] * np.log(rho[m])
    if np.any(~close):
        m = ~close
        result[m] = (1.0 + a_const[m]) * rho[m] + K[m] / (G[m] - 1.0) * rho[m] ** G[m]
    return result


def _compute_integration_constants(Ks, Gammas, rho_boundaries):
    """Integration constants a_i for ε(ρ), anchored at EPS_REF."""
    N_seg = len(Gammas)
    a = np.zeros(N_seg)

    G0, K0 = Gammas[0], Ks[0]
    if abs(G0 - 1.0) < 1e-12:
        a[0] = (EPS_REF - K0 * RHO_REF * np.log(RHO_REF)) / RHO_REF - 1.0
    else:
        a[0] = (EPS_REF - K0 / (G0 - 1.0) * RHO_REF ** G0) / RHO_REF - 1.0

    if not np.isfinite(a[0]):
        return None

    for i in range(1, N_seg):
        rho_b = rho_boundaries[i - 1]
        eps_left = _eps_single(rho_b, Ks[i - 1], Gammas[i - 1], a[i - 1])
        Gi, Ki = Gammas[i], Ks[i]
        if abs(Gi - 1.0) < 1e-12:
            a[i] = (eps_left - Ki * rho_b * np.log(rho_b)) / rho_b - 1.0
        else:
            a[i] = (eps_left - Ki / (Gi - 1.0) * rho_b ** Gi) / rho_b - 1.0
        if not np.isfinite(a[i]):
            return None
    return a


def _evaluate_on_grid(Ks, Gammas, rho_boundaries, a):
    """
    Evaluate c_s^2 = gamma_i P / (epsilon + P) directly on the 200-point ρ-grid.
    Returns (cs2_array, rejection_reason).
    """
    idx = np.searchsorted(rho_boundaries, RHO_GRID)
    G_arr = Gammas[idx]
    K_arr = Ks[idx]
    a_arr = a[idx]
    P_arr = K_arr * RHO_GRID ** G_arr
    eps_arr = _eps_single(RHO_GRID, K_arr, G_arr, a_arr)
    if np.any(eps_arr <= 0):
        return None, Rejection.EPS_NONPOSITIVE
    denom = eps_arr + P_arr
    if np.any(denom <= 0):
        return None, Rejection.EPS_PLUS_P_NONPOSITIVE
    cs2 = G_arr * P_arr / denom
    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, Rejection[tag]
    return cs2, None

def _nudge_from_one(vals):
    """Push any gamma within GAMMA_AVOID of 1.0 to the nearest boundary.
    Rejection-free, deterministic, does not consume RNG state."""
    close = np.abs(vals - 1.0) < GAMMA_AVOID
    vals = vals.copy()
    vals[close & (vals >= 1.0)] = 1.0 + GAMMA_AVOID
    vals[close & (vals < 1.0)] = 1.0 - GAMMA_AVOID
    return vals



_GAMMA_LO = 1.0 - GAMMA_AVOID   # 0.98
_GAMMA_HI = 1.0 + GAMMA_AVOID   # 1.02


def _s_broad_uniform(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = uniform_excluding(GAMMA_MIN, GAMMA_MAX, _GAMMA_LO, _GAMMA_HI, size=N, rng=rng)
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_stiff(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = uniform(1.5, GAMMA_MAX, size=N, rng=rng)  # always > 1.02, safe
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_soft(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = uniform_excluding(GAMMA_MIN, 2.5, _GAMMA_LO, _GAMMA_HI, size=N, rng=rng)
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_peaked(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = _nudge_from_one(one_peaked(N, 1.0, 2.5, 3.0, GAMMA_MAX, rng=rng))
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_beta_centered(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = _nudge_from_one(beta_scaled(2.0, 2.0, GAMMA_MIN, GAMMA_MAX, size=N, rng=rng))
    b = sorted_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_normal_centered(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = _nudge_from_one(normal_truncated(2.5, 1.2, GAMMA_MIN, GAMMA_MAX, size=N, rng=rng))
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_loguniform_gamma(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = _nudge_from_one(log_uniform(GAMMA_MIN, GAMMA_MAX, size=N, rng=rng))
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_alternating(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = _nudge_from_one(alternating_values(N, 2.0, GAMMA_MAX, GAMMA_MIN, 2.0, rng=rng))
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_uniform_nB_boundaries(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = uniform_excluding(GAMMA_MIN, GAMMA_MAX, _GAMMA_LO, _GAMMA_HI, size=N, rng=rng)
    b = sorted_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b

def _s_read_like(rng=None):
    N = pick_count(NSEG_OPTIONS, rng=rng)
    G = uniform(1.5, 5.0, size=N, rng=rng)  # always > 1.02, safe
    b = sorted_log_uniform(N - 1, NB_BOUNDARY_MIN, NB_BOUNDARY_MAX, rng=rng)
    return N, G, b


STRATEGIES = [
    _s_broad_uniform, _s_stiff, _s_soft, _s_peaked,
    _s_beta_centered, _s_normal_centered, _s_loguniform_gamma,
    _s_alternating, _s_uniform_nB_boundaries, _s_read_like,
]

def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    N_seg, Gammas, nB_boundaries = strategy(rng=rng)

    rho_boundaries = m_B * nB_boundaries

    Ks = _fix_K(Gammas, rho_boundaries)
    if Ks is None:
        return None, strategy.__name__, Rejection.K_FAILED

    a = _compute_integration_constants(Ks, Gammas, rho_boundaries)
    if a is None:
        return None, strategy.__name__, Rejection.A_FAILED

    cs2, reason = _evaluate_on_grid(Ks, Gammas, rho_boundaries, a)
    if cs2 is None:
        return None, strategy.__name__, reason

    return cs2, strategy.__name__, None