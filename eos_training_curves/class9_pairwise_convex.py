"""
Class 9: Pairwise convex combinations
"""

from enum import Enum, auto
import numpy as np
from eos_common import cs2_acceptance_check
from sampling_utils import uniform

CLASS_NAME = "Class 9: Pairwise Convex Combinations"
FILE_PREFIX = "class9_pairwise_convex"

_EXCLUDED_PAIRS = frozenset({(5, 6), (5, 7), (6, 7)})


_ALLOWED_PARTNERS = {}
for _i in range(1, 9):
    _partners = []
    for _j in range(1, 9):
        if _j == _i:
            continue
        _pair = tuple(sorted((_i, _j)))
        if _pair not in _EXCLUDED_PAIRS:
            _partners.append(_j)
    _ALLOWED_PARTNERS[_i] = _partners


_ALLOWED_PAIRS = tuple(
    (a, b)
    for a in range(1, 9)
    for b in range(a + 1, 9)
    if (a, b) not in _EXCLUDED_PAIRS
)
assert len(_ALLOWED_PAIRS) == 25, (
    f"Expected 25 allowed pairs, got {len(_ALLOWED_PAIRS)}")
N_ALLOWED_PAIRS = len(_ALLOWED_PAIRS)


_CLASS_LABELS = {
    1: "polytrope",
    2: "spectral",
    3: "cs2_interp",
    4: "RMF",
    5: "CSS",
    6: "quarkyonic",
    7: "NJL_crossover",
    8: "pQCD",
}


class Rejection(Enum):
    PARENT_A_FAILED = auto()
    PARENT_B_FAILED = auto()
    NUMERICAL       = auto()
    ACAUSAL         = auto()
    UNSTABLE        = auto()
    LOW_VARIATION   = auto()


_PARENT_MODULES = None

_PARENT_MODULE_NAMES = [
    "class1_polytrope",
    "class2_spectral",
    "class3_cs2_interpolation",
    "class4_rmf",
    "class5_css",
    "class6_quarkyonic",
    "class7_njl_crossover",
    "class8_pqcd",
]


def _get_parent_modules():
    global _PARENT_MODULES
    if _PARENT_MODULES is None:
        import importlib
        _PARENT_MODULES = [
            importlib.import_module(name) for name in _PARENT_MODULE_NAMES
        ]
    return _PARENT_MODULES


def _draw_parent_curve(module, rng, max_tries=200):
    for _ in range(max_tries):
        result, _, _ = module.generate_one_sample(rng=rng)
        if result is not None:
            return result
    return None
def generate_one_sample(rng=None, parent_cls=None, pair_idx=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    parents = _get_parent_modules()
    if pair_idx is not None:
        cls_A, cls_B = _ALLOWED_PAIRS[int(pair_idx)]
    elif parent_cls is not None:
        cls_A = parent_cls + 1
        partners = _ALLOWED_PARTNERS[cls_A]
        cls_B = partners[int(rng.integers(len(partners)))]
    else:
        cls_A, cls_B = _ALLOWED_PAIRS[int(rng.integers(N_ALLOWED_PAIRS))]

    label_A = _CLASS_LABELS[cls_A]
    label_B = _CLASS_LABELS[cls_B]
    strategy_name = f"_s_pair_{cls_A}_{label_A}_x_{cls_B}_{label_B}"
    cs2_A = _draw_parent_curve(parents[cls_A - 1], rng)
    if cs2_A is None:
        return None, strategy_name, Rejection.PARENT_A_FAILED

    cs2_B = _draw_parent_curve(parents[cls_B - 1], rng)
    if cs2_B is None:
        return None, strategy_name, Rejection.PARENT_B_FAILED
    lam = uniform(0.0, 1.0, rng=rng)
    cs2 = lam * cs2_A + (1.0 - lam) * cs2_B
    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy_name, Rejection[tag]

    return cs2, strategy_name, None