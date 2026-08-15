"""
Class 12 (validation family B): piecewise-linear ("segmented") sound speed.

Construction
------------
c_s^2 is a continuous piecewise-linear function of x = n_B/n_0: random
nodes  x_0 = 0.5 < x_1 < ... < x_{M} < x_{M+1} = 8  carry values
c_i in [0, 1] that are linearly interpolated,

    c_s^2(x) = c_i + (c_{i+1} - c_i) * (x - x_i) / (x_{i+1} - x_i),
                                        x in [x_i, x_{i+1}].

Because a convex combination of numbers in [0,1] stays in [0,1], every
curve satisfies 0 <= c_s^2 <= 1 (causality + stability) by construction.
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

from eos_common import NB_OVER_N0_GRID, cs2_acceptance_check
from sampling_utils import (
    uniform, beta_scaled, sorted_uniform, sorted_log_uniform, pick_count,
)

CLASS_NAME = "Class 12: Piecewise-Linear Sound Speed"
FILE_PREFIX = "class12_pwlinear_cs2"

_X = NB_OVER_N0_GRID.copy()
_L = _X.shape[0]

M_OPTIONS = (1, 2, 3, 4, 5, 6, 7, 8)

XNODE_MIN, XNODE_MAX = 0.6, 7.5      # interior-node window (class-1 style)
CVAL_MIN, CVAL_MAX = 0.0, 1.0        # full causal interval, as in the refs


class Rejection(Enum):
    NODE_ORDER_FAILED = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()          # unreachable by construction; kept for the
    UNSTABLE = auto()         # uniform tag mapping of cs2_acceptance_check
    LOW_VARIATION = auto()


def _interior_positions(M, rng, log_space=False, min_sep=0.05,
                        max_tries=20):
    """Sorted interior node positions with a minimum separation
    (min_sep > grid spacing 0.0377 keeps every segment resolvable)."""
    for _ in range(max_tries):
        pos = (sorted_log_uniform(M, XNODE_MIN, XNODE_MAX, rng=rng)
               if log_space else
               sorted_uniform(M, XNODE_MIN, XNODE_MAX, rng=rng))
        if M <= 1 or np.all(np.diff(pos) >= min_sep):
            return pos
    return None


def _vals(M, rng, lo=CVAL_MIN, hi=CVAL_MAX):
    """M + 2 node values (endpoints included), iid uniform in [lo, hi]."""
    return uniform(lo, hi, size=M + 2, rng=rng)


def _s_broad_uniform(rng=None):
    M = pick_count(M_OPTIONS, rng=rng)
    return _interior_positions(M, rng), _vals(M, rng)


def _s_few_segments(rng=None):
    M = pick_count((1, 2, 3), rng=rng)          # 2-4 segments (Annala-like)
    return _interior_positions(M, rng), _vals(M, rng)


def _s_many_segments(rng=None):
    M = pick_count((6, 7, 8), rng=rng)
    return _interior_positions(M, rng), _vals(M, rng)


def _s_soft_start(rng=None):
    """chi-EFT-like soft value at the 0.5 n0 endpoint."""
    M = pick_count(M_OPTIONS, rng=rng)
    v = _vals(M, rng)
    v[0] = uniform(0.02, 0.20, rng=rng)
    return _interior_positions(M, rng), v


def _s_conformal_end(rng=None):
    """approach c_s^2 ~ 1/3 at the 8 n0 endpoint."""
    M = pick_count(M_OPTIONS, rng=rng)
    v = _vals(M, rng)
    v[-1] = uniform(0.28, 0.40, rng=rng)
    return _interior_positions(M, rng), v


def _s_monotone_rise(rng=None):
    """sorted ascending node values: monotonically stiffening EOS."""
    M = pick_count(M_OPTIONS, rng=rng)
    v = np.sort(_vals(M, rng))
    v[0] = uniform(0.02, 0.25, rng=rng)
    return _interior_positions(M, rng), v


def _s_peaked(rng=None):
    """one high interior node on a low background: a sound-speed bump."""
    M = pick_count((2, 3, 4, 5), rng=rng)
    v = _vals(M, rng, lo=0.0, hi=0.5)
    v[1 + int(rng.integers(M))] = uniform(0.6, 1.0, rng=rng)
    return _interior_positions(M, rng), v


def _s_dipped(rng=None):
    """one low interior node on a moderate background: a V-shaped dip
    (NOT a CSS zero plateau -- no flat segment, no discontinuity)."""
    M = pick_count((2, 3, 4, 5), rng=rng)
    v = _vals(M, rng, lo=0.20, hi=0.80)
    v[1 + int(rng.integers(M))] = uniform(0.0, 0.15, rng=rng)
    return _interior_positions(M, rng), v


def _s_log_positions(rng=None):
    M = pick_count(M_OPTIONS, rng=rng)
    return _interior_positions(M, rng, log_space=True), _vals(M, rng)


def _s_beta_values(rng=None):
    M = pick_count(M_OPTIONS, rng=rng)
    v = beta_scaled(2.0, 2.0, CVAL_MIN, CVAL_MAX, size=M + 2, rng=rng)
    return _interior_positions(M, rng), v


def _s_stiff(rng=None):
    M = pick_count(M_OPTIONS, rng=rng)
    return _interior_positions(M, rng), _vals(M, rng, lo=0.40, hi=1.0)


def _s_soft(rng=None):
    M = pick_count(M_OPTIONS, rng=rng)
    return _interior_positions(M, rng), _vals(M, rng, lo=0.0, hi=0.45)


STRATEGIES = [
    _s_broad_uniform, _s_few_segments, _s_many_segments,
    _s_soft_start, _s_conformal_end, _s_monotone_rise,
    _s_peaked, _s_dipped,
    _s_log_positions, _s_beta_values,
    _s_stiff, _s_soft,
]


def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()

    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    pos, vals = strategy(rng=rng)
    if pos is None:
        return None, strategy.__name__, Rejection.NODE_ORDER_FAILED

    xs = np.concatenate(([_X[0]], np.asarray(pos, dtype=float), [_X[-1]]))
    if np.any(np.diff(xs) <= 0):
        return None, strategy.__name__, Rejection.NODE_ORDER_FAILED

    cs2 = np.interp(_X, xs, np.asarray(vals, dtype=float))

    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]

    return cs2, strategy.__name__, None


if __name__ == "__main__":
    from collections import Counter
    from sampling_utils import set_seed

    rng = set_seed(123)
    N_TEST = 400
    ok, rej, strat = [], Counter(), Counter()
    tries = 0
    while len(ok) < N_TEST and tries < 200 * N_TEST:
        tries += 1
        c, s, r = generate_one_sample(rng=rng)
        if c is None:
            rej[r] += 1
        else:
            ok.append(c)
            strat[s] += 1

    arr = np.array(ok)
    assert arr.shape == (N_TEST, _L), f"bad shape {arr.shape}"
    assert np.all(np.isfinite(arr))
    assert arr.min() >= 0.0 and arr.max() <= 1.0, "causality violated!"
    assert arr.std(axis=1).min() >= 0.01, "low-variation curve slipped in"
    print(f"{CLASS_NAME}")
    print(f"  accepted {len(ok)}/{tries} ({len(ok)/tries:.1%})")
    print(f"  c_s^2 in [{arr.min():.3e}, {arr.max():.6f}]  "
          f"min per-curve std = {arr.std(axis=1).min():.4f}")
    for s, c in strat.most_common():
        print(f"    {s:32s}: {c}")
    for r, c in rej.most_common():
        print(f"    REJ {r.name:24s}: {c}")
    print("  smoke test PASSED")
