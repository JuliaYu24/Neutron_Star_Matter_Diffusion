"""
Class 3: Speed-of-sound model  (logistic + Gaussian)
"""

from enum import Enum, auto
import numpy as np
from eos_common import (
    EPS_REF, P_REF, n0, mN,
    build_pressure_from_cs2, evaluate_cs2_on_grid,
    cs2_acceptance_check,
)
from sampling_utils import (
    uniform, normal_truncated, beta_scaled, log_uniform,
)
EPS_0 = n0 * mN

CLASS_NAME = "Class 3: Speed-of-Sound (Logistic + Gaussian)"
FILE_PREFIX = "class3_cs2_interpolation"

A1_MIN, A1_MAX = 1.0 / 3.0, 1.0
A2_MIN, A2_MAX = 0.05, 2.0
A3_MIN, A3_MAX = 1.5, 37.0
A4_MIN, A4_MAX = 0.1, 1.5
A5_MIN, A5_MAX = 1.5, 12.0
A6_RATIO_MIN, A6_RATIO_MAX = 0.05, 2.0

class Rejection(Enum):
    P_INTEGRATION_FAILED = auto()
    ODE_FAILED = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()
    UNSTABLE = auto()
    LOW_VARIATION = auto()


def _make_cs2_func(a1, a2, a3, a4, a5, a6, extra_bumps=None):
    bumps = [(a4, a5, a6)] + (list(extra_bumps) if extra_bumps else [])
    def cs2(eps):
        x = eps / EPS_0
        out = a1 / (1.0 + np.exp(-a2 * (x - a3)))
        for amp, ctr, wid in bumps:
            out = out + amp * np.exp(-0.5 * ((x - ctr) / wid) ** 2)
        return out
    return cs2

_EPS_COARSE = np.linspace(EPS_REF, 5000.0, 100)

_EPS_MAX_CLASS3 = 5000.0


def _draw_a6(a5, ratio_lo, ratio_hi, rng):
    ratio = uniform(ratio_lo, ratio_hi, rng=rng)
    return ratio * a5


def _s_broad_uniform(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_stiff_plateau(rng=None):
    a1 = uniform(0.6, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_soft_plateau(rng=None):
    a1 = uniform(A1_MIN, 0.5, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_strong_peak(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(0.8, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_weak_peak(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, 0.4, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_early_features(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, 10.0, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, 5.0, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_late_features(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(10.0, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(5.0, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_sharp_peak(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, 0.5, rng)
    return a1, a2, a3, a4, a5, a6


def _s_broad_peak(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, 1.0, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_near_conformal(rng=None):
    a1 = uniform(A1_MIN, 0.45, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, 0.3, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_steep_rise(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(0.6, A2_MAX, rng=rng)
    a3 = uniform(A3_MIN, 8.0, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_gentle_rise(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, 0.3, rng=rng)
    a3 = uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_normal_centered(rng=None):
    a1 = normal_truncated(0.55, 0.15, A1_MIN, A1_MAX, rng=rng)
    a2 = normal_truncated(0.5, 0.25, A2_MIN, A2_MAX, rng=rng)
    a3 = normal_truncated(8.0, 6.0, A3_MIN, A3_MAX, rng=rng)
    a4 = normal_truncated(0.5, 0.3, A4_MIN, A4_MAX, rng=rng)
    a5 = normal_truncated(5.0, 2.5, A5_MIN, A5_MAX, rng=rng)
    ratio = normal_truncated(0.8, 0.4, A6_RATIO_MIN, A6_RATIO_MAX, rng=rng)
    a6 = ratio * a5
    return a1, a2, a3, a4, a5, a6


def _s_beta_centered(rng=None):
    a1 = beta_scaled(2.0, 2.0, A1_MIN, A1_MAX, rng=rng)
    a2 = beta_scaled(2.0, 2.0, A2_MIN, A2_MAX, rng=rng)
    a3 = beta_scaled(2.0, 2.0, A3_MIN, A3_MAX, rng=rng)
    a4 = beta_scaled(2.0, 2.0, A4_MIN, A4_MAX, rng=rng)
    a5 = beta_scaled(2.0, 2.0, A5_MIN, A5_MAX, rng=rng)
    ratio = beta_scaled(2.0, 2.0, A6_RATIO_MIN, A6_RATIO_MAX, rng=rng)
    a6 = ratio * a5
    return a1, a2, a3, a4, a5, a6


def _s_log_uniform_positions(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a3 = log_uniform(A3_MIN, A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a5 = log_uniform(A5_MIN, A5_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


def _s_peak_before_plateau(rng=None):
    a1 = uniform(A1_MIN, A1_MAX, rng=rng)
    a2 = uniform(A2_MIN, A2_MAX, rng=rng)
    a5 = uniform(A5_MIN, A5_MAX, rng=rng)
    # Ensure a3 > a5 so the logistic turns on later
    a3 = uniform(max(A3_MIN, a5 + 1.0), A3_MAX, rng=rng)
    a4 = uniform(A4_MIN, A4_MAX, rng=rng)
    a6 = _draw_a6(a5, A6_RATIO_MIN, A6_RATIO_MAX, rng)
    return a1, a2, a3, a4, a5, a6


STRATEGIES = [
    _s_broad_uniform, _s_stiff_plateau, _s_soft_plateau,
    _s_strong_peak, _s_weak_peak,
    _s_early_features, _s_late_features,
    _s_sharp_peak, _s_broad_peak,
    _s_near_conformal, _s_steep_rise, _s_gentle_rise,
    _s_normal_centered, _s_beta_centered,
    _s_log_uniform_positions, _s_peak_before_plateau,
]



def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()
    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    a1, a2, a3, a4, a5, a6 = strategy(rng=rng)
    extra_bumps = None
    if rng.uniform() < 0.35:
        a4b = uniform(A4_MIN, A4_MAX, rng=rng)
        a5b = uniform(A5_MIN, A5_MAX, rng=rng)
        a6b = _draw_a6(a5b, A6_RATIO_MIN, A6_RATIO_MAX, rng)
        extra_bumps = [(a4b, a5b, a6b)]
    cs2_func = _make_cs2_func(a1, a2, a3, a4, a5, a6, extra_bumps=extra_bumps)
    cs2_coarse = cs2_func(_EPS_COARSE)
    if np.any(cs2_coarse > 1.0):
        return None, strategy.__name__, Rejection.ACAUSAL
    P_func = build_pressure_from_cs2(cs2_func, eps_max=_EPS_MAX_CLASS3)
    if P_func is None:
        return None, strategy.__name__, Rejection.P_INTEGRATION_FAILED
    cs2 = evaluate_cs2_on_grid(P_func, cs2_func, eps_max=_EPS_MAX_CLASS3)
    if cs2 is None:
        return None, strategy.__name__, Rejection.ODE_FAILED
    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]
    return cs2, strategy.__name__, None