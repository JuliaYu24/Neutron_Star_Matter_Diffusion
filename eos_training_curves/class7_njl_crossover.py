"""
Class 7: NJL : quark–meson-inspired crossover model
"""

from enum import Enum, auto
import numpy as np
from eos_common import n0, NB_OVER_N0_GRID, cs2_acceptance_check
from sampling_utils import (
    uniform, beta_scaled, normal_truncated, log_uniform,
)

CLASS_NAME = "Class 7: NJL/quark–meson crossover"
FILE_PREFIX = "class7_njl_crossover"


A_MIN, A_MAX = 0.05, 0.60
NCHI_MIN, NCHI_MAX = 1.5, 5.0
SIGMA_MIN, SIGMA_MAX = 0.3, 3.0
GAMMA1_MIN, GAMMA1_MAX = 3, 10
GAMMA2_MIN, GAMMA2_MAX = 2, 8

class Rejection(Enum):
    NUMERICAL     = auto()
    ACAUSAL       = auto()
    UNSTABLE      = auto()
    LOW_VARIATION = auto()

def _cs2_profile(A, nchi, sigma, gamma1, gamma2, extra_peaks=None):
    nbar = NB_OVER_N0_GRID
    nbar_g1 = nbar ** gamma1
    conformal = (1.0 / 3.0) * nbar_g1 / (1.0 + nbar_g1)
    nbar_g2 = nbar ** gamma2
    onset2 = nbar_g2 / (1.0 + nbar_g2)
    peaks = [(A, nchi, sigma)] + (list(extra_peaks) if extra_peaks else [])
    peak = np.zeros_like(nbar)
    for amp, ctr, wid in peaks:
        peak = peak + amp * np.exp(-0.5 * ((nbar - ctr) / wid) ** 2) * onset2
    return conformal + peak
def _s_broad_uniform(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_strong_peak(rng=None):
    return (uniform(0.25, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_weak_peak(rng=None):
    return (uniform(A_MIN, 0.15, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_early_crossover(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, 2.5, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_late_crossover(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(3.0, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_sharp_crossover(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, 1.0, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_broad_crossover(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(1.5, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_steep_onset(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(6, GAMMA1_MAX, rng=rng),
            uniform(4, GAMMA2_MAX, rng=rng))


def _s_gentle_onset(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, 5, rng=rng),
            uniform(GAMMA2_MIN, 3, rng=rng))


def _s_beta_centered(rng=None):
    return (beta_scaled(2.0, 2.0, A_MIN, A_MAX, rng=rng),
            beta_scaled(2.0, 2.0, NCHI_MIN, NCHI_MAX, rng=rng),
            beta_scaled(2.0, 2.0, SIGMA_MIN, SIGMA_MAX, rng=rng),
            beta_scaled(2.0, 2.0, GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            beta_scaled(2.0, 2.0, GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_normal_centered(rng=None):
    return (normal_truncated(0.25, 0.10, A_MIN, A_MAX, rng=rng),
            normal_truncated(2.5, 0.7, NCHI_MIN, NCHI_MAX, rng=rng),
            normal_truncated(1.5, 0.6, SIGMA_MIN, SIGMA_MAX, rng=rng),
            normal_truncated(6.0, 1.0, GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            normal_truncated(4.0, 1.0, GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_strong_sharp(rng=None):
    return (uniform(0.25, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            uniform(SIGMA_MIN, 1.2, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


def _s_loguniform_width(rng=None):
    return (uniform(A_MIN, A_MAX, rng=rng),
            uniform(NCHI_MIN, NCHI_MAX, rng=rng),
            log_uniform(SIGMA_MIN, SIGMA_MAX, rng=rng),
            uniform(GAMMA1_MIN, GAMMA1_MAX, rng=rng),
            uniform(GAMMA2_MIN, GAMMA2_MAX, rng=rng))


STRATEGIES = [
    _s_broad_uniform,
    _s_strong_peak,
    _s_weak_peak,
    _s_early_crossover,
    _s_late_crossover,
    _s_sharp_crossover,
    _s_broad_crossover,
    _s_steep_onset,
    _s_gentle_onset,
    _s_beta_centered,
    _s_normal_centered,
    _s_strong_sharp,
    _s_loguniform_width,
]

def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    A, nchi, sigma, gamma1, gamma2 = strategy(rng=rng)
    extra_peaks = None
    if rng.uniform() < 0.30:
        A2 = uniform(A_MIN, A_MAX, rng=rng)
        nchi2 = uniform(NCHI_MIN, NCHI_MAX, rng=rng)
        sigma2 = uniform(SIGMA_MIN, SIGMA_MAX, rng=rng)
        extra_peaks = [(A2, nchi2, sigma2)]

    cs2 = _cs2_profile(A, nchi, sigma, gamma1, gamma2, extra_peaks=extra_peaks)

    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]

    return cs2, strategy.__name__, None
