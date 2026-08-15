"""
Class 5: Constant Speed of Sound (CSS) with Phase Transition
"""

from enum import Enum, auto
import numpy as np
from eos_common import (
    EPS_REF, P_REF, n0, NB_GRID, N_GRID,
    cs2_acceptance_check, thermodynamic_check,
)
from sampling_utils import uniform

CLASS_NAME = "Class 5: CSS Phase Transition"
FILE_PREFIX = "class5_css"


NTRANS_MIN, NTRANS_MAX = 1.5, 7.0
DEPS_RATIO_MIN, DEPS_RATIO_MAX = 0.01, 1.5
C2QM_MIN, C2QM_MAX = 0.05, 1.0

class Rejection(Enum):
    PARENT_INTEGRATION_FAILED = auto()
    NQ_OFF_GRID = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()
    UNSTABLE = auto()
    LOW_VARIATION = auto()

_PARENT_MODULES = None

_PARENT_MODULE_NAMES = [
    "class1_polytrope",
    "class2_spectral",
    "class3_cs2_interpolation",
    "class4_rmf",
]


def _get_parent_modules():
    global _PARENT_MODULES
    if _PARENT_MODULES is None:
        import importlib
        _PARENT_MODULES = [
            importlib.import_module(name) for name in _PARENT_MODULE_NAMES
        ]
    return _PARENT_MODULES

def _integrate_parent(cs2_on_grid):

    reason, eps_arr, P_arr = thermodynamic_check(
        cs2_on_grid,
        max_P_over_eps=1e10,
        min_eps_per_baryon=0.0,
        max_eps_per_baryon=1e10,
    )
    if eps_arr is None:
        return None, None
    return eps_arr, P_arr

def _interpolate_at_density(nB_target, arr):
    idx = np.searchsorted(NB_GRID, nB_target) - 1
    idx = np.clip(idx, 0, N_GRID - 2)
    frac = (nB_target - NB_GRID[idx]) / (NB_GRID[idx + 1] - NB_GRID[idx])
    return arr[idx] + frac * (arr[idx + 1] - arr[idx])

def _draw_css_params(rng, n_trans_range=None, deps_range=None, c2qm_range=None):
    nt_lo, nt_hi = n_trans_range or (NTRANS_MIN, NTRANS_MAX)
    de_lo, de_hi = deps_range or (DEPS_RATIO_MIN, DEPS_RATIO_MAX)
    cq_lo, cq_hi = c2qm_range or (C2QM_MIN, C2QM_MAX)
    return (uniform(nt_lo, nt_hi, rng=rng),
            uniform(de_lo, de_hi, rng=rng),
            uniform(cq_lo, cq_hi, rng=rng))


def _s_broad_uniform(rng=None):
    return _draw_css_params(rng)


def _s_early_strong(rng=None):
    return _draw_css_params(rng, n_trans_range=(1.5, 3.0), deps_range=(0.3, DEPS_RATIO_MAX))


def _s_late_weak(rng=None):
    return _draw_css_params(rng, n_trans_range=(4.0, NTRANS_MAX), deps_range=(DEPS_RATIO_MIN, 0.3))


def _s_stiff_quark(rng=None):
    return _draw_css_params(rng, c2qm_range=(0.5, C2QM_MAX))


def _s_soft_quark(rng=None):
    return _draw_css_params(rng, c2qm_range=(C2QM_MIN, 0.33))


def _s_twin_star(rng=None):
    return _draw_css_params(rng, deps_range=(0.5, DEPS_RATIO_MAX), c2qm_range=(0.3, 0.8))


def _s_conformal(rng=None):
    return _draw_css_params(rng, c2qm_range=(0.30, 0.36))


STRATEGIES = [
    _s_broad_uniform, _s_early_strong, _s_late_weak,
    _s_stiff_quark, _s_soft_quark, _s_twin_star, _s_conformal,
]


def generate_one_sample(rng=None, parent_cls=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()
    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    n_trans_over_n0, deps_ratio, c2_qm = strategy(rng=rng)

    parents = _get_parent_modules()
    cls_idx = parent_cls if parent_cls is not None else rng.integers(4)
    parent_mod = parents[cls_idx]
    cs2_parent = None
    for _ in range(200):
        result, _, reason = parent_mod.generate_one_sample(rng=rng)
        if result is not None:
            cs2_parent = result
            break

    if cs2_parent is None:
        return None, strategy.__name__, Rejection.PARENT_INTEGRATION_FAILED

    eps_arr, P_arr = _integrate_parent(cs2_parent)
    if eps_arr is None:
        return None, strategy.__name__, Rejection.PARENT_INTEGRATION_FAILED

    n_trans = n_trans_over_n0 * n0
    eps_trans = _interpolate_at_density(n_trans, eps_arr)
    P_trans = _interpolate_at_density(n_trans, P_arr)

    delta_eps = deps_ratio * eps_trans
    nQ_B = n_trans * (eps_trans + delta_eps + P_trans) / (eps_trans + P_trans)

    if nQ_B > NB_GRID[-1]:
        return None, strategy.__name__, Rejection.NQ_OFF_GRID

    cs2 = np.where(
        NB_GRID < n_trans,
        cs2_parent,
        np.where(NB_GRID <= nQ_B, 0.0, c2_qm),
    )
    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]

    return cs2, strategy.__name__, None
