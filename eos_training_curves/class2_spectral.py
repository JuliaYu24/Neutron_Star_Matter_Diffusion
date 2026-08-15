"""

Class 2: Spectral Representation: log Gamma(P) = sum_k gamma_k (ln p /p_0)^k
"""

from enum import Enum, auto
import numpy as np
from eos_common import P_REF, evaluate_cs2_on_grid_P, cs2_acceptance_check
from sampling_utils import (
    uniform, pick_count, normal_truncated, beta_scaled,
)

P0 = P_REF

CLASS_NAME = "Class 2: Spectral Representation"
FILE_PREFIX = "class2_spectral"


DEFAULT_RANGES = [
    (0.5,   3.5),
    (-0.20, 0.20),
    (-0.015, 0.015),
    (-0.002, 0.002),
    (-0.0002, 0.0002),
    (-3e-5, 3e-5),
    (-4e-6, 4e-6),
    (-5e-7, 5e-7),
    (-6e-8, 6e-8),
    (-8e-9, 8e-9),
]
K_OPTIONS = (2, 3, 4, 5, 6, 7, 8, 9)


class Rejection(Enum):
    ODE_FAILED = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()
    UNSTABLE = auto()
    LOW_VARIATION = auto()

def _make_Gamma_func(gamma_coeffs):
    coeffs = tuple(gamma_coeffs)

    def Gamma(P):
        x = np.log(P / P0)
        log_Gamma = 0.0
        xk = 1.0
        for gk in coeffs:
            log_Gamma += gk * xk
            xk *= x
        return np.exp(log_Gamma)

    return Gamma


def _sample_coeffs(K, rng, ranges=None):
    if ranges is None:
        ranges = DEFAULT_RANGES
    return np.array([uniform(ranges[k][0], ranges[k][1], rng=rng)
                     for k in range(K + 1)])


def _s_broad_uniform(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    return _sample_coeffs(K, rng)

def _s_low_order(rng=None):
    return _sample_coeffs(2, rng)

def _s_high_order(rng=None):
    return _sample_coeffs(9, rng)

def _s_stiff(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    c = _sample_coeffs(K, rng)
    c[0] = uniform(1.5, 3.5, rng=rng)
    return c

def _s_soft(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    c = _sample_coeffs(K, rng)
    c[0] = uniform(0.5, 1.2, rng=rng)
    return c

def _s_wiggly(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    wide = [
        (0.5,   3.5),
        (-0.35, 0.35),
        (-0.025, 0.025),
        (-0.003, 0.003),
        (-0.0004, 0.0004),
        (-5e-5, 5e-5),
        (-7e-6, 7e-6),
        (-9e-7, 9e-7),
        (-1e-7, 1e-7),
        (-1.4e-8, 1.4e-8),
    ]
    return _sample_coeffs(K, rng, ranges=wide)

def _s_normal_centered(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    mus  = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sigs = [0.6, 0.10, 0.007, 0.001, 0.0001, 1.5e-5, 2e-6, 2.5e-7, 3e-8, 4e-9]
    return np.array([
        normal_truncated(mus[k], sigs[k],
                         DEFAULT_RANGES[k][0], DEFAULT_RANGES[k][1],
                         rng=rng)
        for k in range(K + 1)
    ])

def _s_beta_centered(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    return np.array([
        beta_scaled(2.0, 2.0,
                    DEFAULT_RANGES[k][0], DEFAULT_RANGES[k][1],
                    rng=rng)
        for k in range(K + 1)
    ])

def _s_lindblom_like(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    narrow = [
        (0.5,   2.0),
        (-0.15, 0.15),
        (-0.010, 0.010),
        (-0.0015, 0.0015),
        (-0.0002, 0.0002),
        (-2e-5, 2e-5),
        (-3e-6, 3e-6),
        (-4e-7, 4e-7),
        (-5e-8, 5e-8),
        (-6e-9, 6e-9),
    ]
    return _sample_coeffs(K, rng, ranges=narrow)

def _s_large_slope(rng=None):
    K = pick_count(K_OPTIONS, rng=rng)
    c = _sample_coeffs(K, rng)
    c[1] = uniform(-0.35, 0.35, rng=rng)
    return c


STRATEGIES = [
    _s_broad_uniform, _s_low_order, _s_high_order,
    _s_stiff, _s_soft, _s_wiggly,
    _s_normal_centered, _s_beta_centered,
    _s_lindblom_like, _s_large_slope,
]


def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    gamma_coeffs = strategy(rng=rng)

    Gamma_func = _make_Gamma_func(gamma_coeffs)
    cs2 = evaluate_cs2_on_grid_P(Gamma_func)

    if cs2 is None:
        return None, strategy.__name__, Rejection.ODE_FAILED

    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]

    return cs2, strategy.__name__, None
