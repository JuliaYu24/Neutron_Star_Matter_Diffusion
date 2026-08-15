"""
Generic random sampling utilities reusable across all EOS functional classes.
"""

import numpy as np

_rng = np.random.default_rng()


def set_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng


def get_rng():
    return _rng


def _r(rng):
    return rng if rng is not None else _rng

def uniform(lo, hi, size=None, rng=None):
    return _r(rng).uniform(lo, hi, size=size)


def log_uniform(lo, hi, size=None, rng=None):
    return np.exp(_r(rng).uniform(np.log(lo), np.log(hi), size=size))


def beta_scaled(a, b, lo, hi, size=None, rng=None):
    return lo + _r(rng).beta(a, b, size=size) * (hi - lo)


def normal_truncated(mu, sigma, lo, hi, size=None, rng=None):
    vals = _r(rng).normal(mu, sigma, size=size)
    return np.clip(vals, lo, hi)

def sorted_uniform(n, lo, hi, rng=None):
    if n <= 0:
        return np.array([])
    return np.sort(_r(rng).uniform(lo, hi, size=n))


def sorted_log_uniform(n, lo, hi, rng=None):
    if n <= 0:
        return np.array([])
    return np.sort(log_uniform(lo, hi, size=n, rng=rng))


def sorted_via_interp(n, lo, hi, interp_func, log_space=False, rng=None):
    if n <= 0:
        return np.array([])
    if log_space:
        vals = sorted_log_uniform(n, lo, hi, rng=rng)
    else:
        vals = sorted_uniform(n, lo, hi, rng=rng)
    return interp_func(vals)

def pick_count(options, rng=None):
    return int(_r(rng).choice(options))


def alternating_values(n, lo_even, hi_even, lo_odd, hi_odd, rng=None):
    r = _r(rng)
    vals = np.zeros(n)
    for i in range(n):
        if i % 2 == 0:
            vals[i] = r.uniform(lo_even, hi_even)
        else:
            vals[i] = r.uniform(lo_odd, hi_odd)
    return vals


def one_peaked(n, base_lo, base_hi, peak_lo, peak_hi, rng=None):
    r = _r(rng)
    vals = r.uniform(base_lo, base_hi, size=n)
    vals[r.integers(n)] = r.uniform(peak_lo, peak_hi)
    return vals


def uniform_excluding(lo, hi, excl_lo, excl_hi, size=None, rng=None):
    r = _r(rng)
    w_left = max(excl_lo - lo, 0.0)
    w_right = max(hi - excl_hi, 0.0)
    w_total = w_left + w_right
    if w_total <= 0:
        raise ValueError("Exclusion zone covers entire range")

    n_out = 1 if size is None else size
    vals = np.empty(n_out)
    choose_left = r.uniform(size=n_out) < (w_left / w_total)
    n_left = int(np.sum(choose_left))
    if n_left > 0:
        vals[choose_left] = r.uniform(lo, excl_lo, size=n_left)
    if n_left < n_out:
        vals[~choose_left] = r.uniform(excl_hi, hi, size=n_out - n_left)

    if size is None:
        return float(vals[0])
    return vals
