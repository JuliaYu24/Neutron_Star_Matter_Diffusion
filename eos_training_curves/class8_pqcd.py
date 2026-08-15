"""
Class 8: pQCD-motivated high-density profiles
"""

from enum import Enum, auto
import numpy as np
from eos_common import n0, NB_OVER_N0_GRID, cs2_acceptance_check
from sampling_utils import (
    uniform, log_uniform, beta_scaled, normal_truncated,
)
CLASS_NAME = "Class 8: pQCD-motivated high-density"
FILE_PREFIX = "class8_pqcd"

GAMMA = 4
GAMMA_MIN, GAMMA_MAX = 2.0, 8.0

A_MIN,      A_MAX      = 0.0,  0.60
NP_MIN,     NP_MAX     = 2.0,  5.0
SIGP_MIN,   SIGP_MAX   = 0.3,  3.0
DELTA_MIN,  DELTA_MAX  = 0.02, 0.25
NDEL_MIN,   NDEL_MAX   = 1.5,  4.0
ALPHA_MIN,  ALPHA_MAX  = 1.0,  6.0


class Rejection(Enum):
    NUMERICAL     = auto()
    ACAUSAL       = auto()
    UNSTABLE      = auto()
    LOW_VARIATION = auto()


def _cs2_profile(nbar, A, np_, sigp, delta_inf, ndel, alpha, gamma=GAMMA):
    gaussian = A * np.exp(-0.5 * ((nbar - np_) / sigp) ** 2)
    ratio = np.where(nbar > 0, ndel / nbar, 1e30)
    deficit = delta_inf / (1.0 + ratio ** alpha)
    bracket = 1.0 / 3.0 + gaussian - deficit
    onset = nbar ** gamma / (1.0 + nbar ** gamma)

    return bracket * onset


def _evaluate_on_grid(A, np_, sigp, delta_inf, ndel, alpha, gamma=GAMMA):
    cs2 = _cs2_profile(NB_OVER_N0_GRID, A, np_, sigp, delta_inf, ndel, alpha, gamma=gamma)
    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, Rejection[tag]
    return cs2, None

def _s_broad_uniform(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_strong_peak(rng=None):
    return (uniform(0.30, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, 1.8, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_no_peak(rng=None):
    return (uniform(A_MIN, 0.05, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_large_deficit(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(0.10, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_small_deficit(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, 0.05, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_early_deficit(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, 2.5, rng=rng),
            uniform(ALPHA_MIN, 2.5, rng=rng))


def _s_late_deficit(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(3.0, NDEL_MAX, rng=rng),
            uniform(2.5, ALPHA_MAX, rng=rng))


def _s_early_peak(rng=None):
    return (uniform(0.10, A_MAX, rng=rng),
            uniform(NP_MIN, 3.0, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_late_peak(rng=None):
    return (uniform(0.10, A_MAX, rng=rng),
            uniform(4.0, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_narrow_peak(rng=None):
    return (uniform(0.15, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, 1.0, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_wide_peak(rng=None):
    return (uniform(0.05, 0.35, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(1.8, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_peak_before_deficit(rng=None):
    np_ = uniform(NP_MIN, 3.5, rng=rng)
    ndel = uniform(max(np_ + 0.5, NDEL_MIN), NDEL_MAX, rng=rng)
    return (uniform(0.10, A_MAX, rng=rng),
            np_,
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            ndel,
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_overlapping(rng=None):
    centre = uniform(2.0, 4.0, rng=rng)
    return (uniform(0.10, A_MAX, rng=rng),
            centre,
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(max(centre - 1.0, NDEL_MIN),
                    min(centre + 1.0, NDEL_MAX), rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_beta_centered(rng=None):
    return (beta_scaled(2.0, 2.0, A_MIN, A_MAX, rng=rng),
            beta_scaled(2.0, 2.0, NP_MIN, NP_MAX, rng=rng),
            beta_scaled(2.0, 2.0, SIGP_MIN, SIGP_MAX, rng=rng),
            beta_scaled(2.0, 2.0, DELTA_MIN, DELTA_MAX, rng=rng),
            beta_scaled(2.0, 2.0, NDEL_MIN, NDEL_MAX, rng=rng),
            beta_scaled(2.0, 2.0, ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_normal_centered(rng=None):
    return (normal_truncated(0.25, 0.15, A_MIN, A_MAX, rng=rng),
            normal_truncated(3.5, 1.0, NP_MIN, NP_MAX, rng=rng),
            normal_truncated(1.5, 0.7, SIGP_MIN, SIGP_MAX, rng=rng),
            normal_truncated(0.08, 0.04, DELTA_MIN, DELTA_MAX, rng=rng),
            normal_truncated(2.5, 0.8, NDEL_MIN, NDEL_MAX, rng=rng),
            normal_truncated(2.5, 1.0, ALPHA_MIN, ALPHA_MAX, rng=rng))


def _s_log_uniform_delta(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NP_MIN, NP_MAX, rng=rng),
            uniform(SIGP_MIN, SIGP_MAX, rng=rng),
            log_uniform(DELTA_MIN, DELTA_MAX, rng=rng),
            uniform(NDEL_MIN, NDEL_MAX, rng=rng),
            uniform(ALPHA_MIN, ALPHA_MAX, rng=rng))


STRATEGIES = [
    _s_broad_uniform,
    _s_strong_peak,
    _s_no_peak,
    _s_large_deficit,
    _s_small_deficit,
    _s_early_deficit,
    _s_late_deficit,
    _s_early_peak,
    _s_late_peak,
    _s_narrow_peak,
    _s_wide_peak,
    _s_peak_before_deficit,
    _s_overlapping,
    _s_beta_centered,
    _s_normal_centered,
    _s_log_uniform_delta,
]


def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    A, np_, sigp, delta_inf, ndel, alpha = strategy(rng=rng)
    gamma = uniform(GAMMA_MIN, GAMMA_MAX, rng=rng)

    cs2, reason = _evaluate_on_grid(A, np_, sigp, delta_inf, ndel, alpha, gamma=gamma)
    if cs2 is None:
        return None, strategy.__name__, reason

    return cs2, strategy.__name__, None
