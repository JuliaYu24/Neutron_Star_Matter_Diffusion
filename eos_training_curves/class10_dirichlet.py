"""
Class 10: Dirichlet mixtures

For each draw, M in {3, 4, 5} parent families are chosen at random
(with M itself drawn uniformly) and their c^2_s profiles are combined
with Dirichlet-distributed weights

    (lambda_1, ..., lambda_M) ~ Dir(alpha, ..., alpha),

where the concentration parameter alpha is itself randomised
(log-uniformly on [0.1, 10]) to interpolate between near-uniform
and peaked weightings.

Thermodynamic consistency is guaranteed by construction: since
0 <= c^2_s^i <= 1 for every parent, the Dirichlet-weighted sum
satisfies the same pointwise bounds for any weight vector on the
simplex.
"""

from enum import Enum, auto
import numpy as np
from eos_common import cs2_acceptance_check
from sampling_utils import log_uniform, uniform


CLASS_NAME = "Class 10: Dirichlet Mixtures"
FILE_PREFIX = "class10_dirichlet"


_QUARK_ONSET_CLASSES = {5, 6, 7}
_SAFE_CLASSES = [1, 2, 3, 4, 8]
_ALL_CLASSES = list(range(1, 9))

ALPHA_MIN, ALPHA_MAX = 0.1, 10.0
M_OPTIONS = (3, 4, 5)

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
    PARENT_FAILED   = auto()
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



def _enforce_quark_onset_constraint(labels, rng):
    labels = list(labels)
    quark_indices = [i for i, c in enumerate(labels) if c in _QUARK_ONSET_CLASSES]

    if len(quark_indices) <= 1:
        return labels

    # Keep one at random, redraw the rest
    keep = quark_indices[int(rng.integers(len(quark_indices)))]
    for idx in quark_indices:
        if idx != keep:
            labels[idx] = _SAFE_CLASSES[int(rng.integers(len(_SAFE_CLASSES)))]

    return labels


def _draw_class_labels(M, rng, pool=None):
    if pool is None:
        pool = _ALL_CLASSES
    return [pool[int(rng.integers(len(pool)))] for _ in range(M)]


def _s_broad_baseline(rng=None):
    M = int(rng.choice(M_OPTIONS))
    alpha = log_uniform(ALPHA_MIN, ALPHA_MAX, rng=rng)
    labels = _draw_class_labels(M, rng)
    labels = _enforce_quark_onset_constraint(labels, rng)
    return M, float(alpha), labels


def _s_peaked_weights(rng=None):
    M = int(rng.choice(M_OPTIONS))
    alpha = log_uniform(0.1, 0.5, rng=rng)
    labels = _draw_class_labels(M, rng)
    labels = _enforce_quark_onset_constraint(labels, rng)
    return M, float(alpha), labels


def _s_uniform_weights(rng=None):
    M = int(rng.choice(M_OPTIONS))
    alpha = log_uniform(3.0, ALPHA_MAX, rng=rng)
    labels = _draw_class_labels(M, rng)
    labels = _enforce_quark_onset_constraint(labels, rng)
    return M, float(alpha), labels



def _s_hadronic_only(rng=None):
    M = int(rng.choice(M_OPTIONS))
    alpha = log_uniform(ALPHA_MIN, ALPHA_MAX, rng=rng)
    labels = _draw_class_labels(M, rng, pool=[1, 2, 3, 4])
    return M, float(alpha), labels


def _s_with_quark_onset(rng=None):
    M = int(rng.choice(M_OPTIONS))
    alpha = log_uniform(ALPHA_MIN, ALPHA_MAX, rng=rng)
    quark_cls = [int(rng.choice(list(_QUARK_ONSET_CLASSES)))]
    others = _draw_class_labels(M - 1, rng, pool=_SAFE_CLASSES)
    labels = quark_cls + others
    rng.shuffle(labels)
    return M, float(alpha), list(labels)


def _s_with_pqcd(rng=None):
    M = int(rng.choice(M_OPTIONS))
    alpha = log_uniform(ALPHA_MIN, ALPHA_MAX, rng=rng)
    labels = [8] + _draw_class_labels(M - 1, rng)
    labels = _enforce_quark_onset_constraint(labels, rng)
    rng.shuffle(labels)
    return M, float(alpha), list(labels)


STRATEGIES = [
    _s_broad_baseline,
    _s_peaked_weights,
    _s_uniform_weights,
    _s_hadronic_only,
    _s_with_quark_onset,
    _s_with_pqcd,
]

def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    parents = _get_parent_modules()

    strategy = STRATEGIES[int(rng.integers(len(STRATEGIES)))]
    M, alpha, class_labels = strategy(rng=rng)

    label_str = "_".join(str(c) for c in sorted(class_labels))
    strategy_name = (f"{strategy.__name__}_M{M}_a{alpha:.2f}"
                     f"_cls{label_str}")

    weights = rng.dirichlet(np.full(M, alpha))

    cs2_parents = []
    for cls_label in class_labels:
        cs2_i = _draw_parent_curve(parents[cls_label - 1], rng)
        if cs2_i is None:
            return None, strategy_name, Rejection.PARENT_FAILED
        cs2_parents.append(cs2_i)

    cs2 = np.zeros_like(cs2_parents[0])
    for w, cs2_i in zip(weights, cs2_parents):
        cs2 += w * cs2_i
    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy_name, Rejection[tag]

    return cs2, strategy_name, None