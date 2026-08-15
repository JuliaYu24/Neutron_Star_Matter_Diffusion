"""
Class 6: Quarkyonic Matter:  McLerran–Reddy model
"""

from enum import Enum, auto
import numpy as np
from scipy.optimize import brentq
from eos_common import (
    n0, mN, NB_GRID, N_GRID,
    hc, hc3,
    fermi_energy_integral, kF_from_density,
    cs2_from_eps_array, cs2_acceptance_check,
)
from sampling_utils import (
    uniform, normal_truncated, beta_scaled, log_uniform,
)

NC = 3
MQ = mN / NC

CLASS_NAME = "Class 6: Quarkyonic Matter (McLerran–Reddy)"
FILE_PREFIX = "class6_quarkyonic"

LAMBDA_MIN, LAMBDA_MAX = 150.0, 600.0
NT_OVER_N0_MIN, NT_OVER_N0_MAX = 1.0, 5.0
A_MIN, A_MAX = -40.0, 20.0
ALPHA_MIN, ALPHA_MAX = 0.3, 1.5
B_MIN, B_MAX = 0.0, 30.0
BETA_MIN, BETA_MAX = 1.5, 3.5


class Rejection(Enum):
    KTN_LE_LAMBDA = auto()
    ROOT_FIND_FAILED = auto()
    EPS_NONPOSITIVE = auto()
    P_NONPOSITIVE = auto()
    DE_NONPOSITIVE = auto()
    NUMERICAL = auto()
    ACAUSAL = auto()
    UNSTABLE = auto()
    LOW_VARIATION = auto()


def _compute_kappa(Lambda, ktn):
    return NC**2 * (ktn / Lambda - Lambda**2 / ktn**2)


def _shell_quantities(kFn, Lambda, kappa):
    Delta = Lambda**3 / kFn**2 + kappa * Lambda / NC**2
    k0n = max(kFn - Delta, 0.0)
    kFd = k0n / NC
    kFu = kFd / 2.0**(1.0 / 3.0)
    nn = (kFn**3 - k0n**3) / (3.0 * np.pi**2 * hc3)
    return k0n, kFd, kFu, nn


def _total_baryon_number(kFn, Lambda, kappa):
    k0n, kFd, kFu, _ = _shell_quantities(kFn, Lambda, kappa)
    return (kFn**3 - k0n**3 + kFd**3 + kFu**3) / (3.0 * np.pi**2 * hc3)


def _Vn(nn, a, alpha, b, beta):
    x = nn / n0
    return a * x**alpha + b * x**beta

def _energy_density_below(nB, a, alpha, b, beta):
    kFn = kF_from_density(nB)
    eps_kin = fermi_energy_integral(kFn, mN) / (np.pi**2 * hc3)
    eps_int = nB * _Vn(nB, a, alpha, b, beta)
    return eps_kin + eps_int


def _energy_density_above(kFn, k0n, kFd, kFu, nn,
                          a, alpha, b, beta):
    eps_n_kin = (fermi_energy_integral(kFn, mN)
                 - fermi_energy_integral(k0n, mN)) / (np.pi**2 * hc3)
    eps_n_int = nn * _Vn(nn, a, alpha, b, beta)
    eps_q_kin = NC * (fermi_energy_integral(kFd, MQ)
                      + fermi_energy_integral(kFu, MQ)) / (np.pi**2 * hc3)
    return eps_n_kin + eps_n_int + eps_q_kin

def _find_kFn(nB_target, Lambda, kappa, ktn):
    def residual(kFn):
        return _total_baryon_number(kFn, Lambda, kappa) - nB_target
    kFn_max = 2.0 * kF_from_density(nB_target)
    for _ in range(5):
        if residual(kFn_max) > 0:
            break
        kFn_max *= 1.5
    else:
        return None

    try:
        return brentq(residual, ktn, kFn_max, xtol=1e-6, rtol=1e-10)
    except (ValueError, RuntimeError):
        return None

def _evaluate_on_grid(Lambda, nt_over_n0, a, alpha, b, beta):
    nt = nt_over_n0 * n0
    ktn = kF_from_density(nt)
    if ktn <= Lambda:
        return None, Rejection.KTN_LE_LAMBDA

    kappa = _compute_kappa(Lambda, ktn)

    eps_arr = np.empty(N_GRID)

    for i in range(N_GRID):
        nB = NB_GRID[i]

        if nB < nt:
            eps_arr[i] = _energy_density_below(nB, a, alpha, b, beta)
        else:
            kFn = _find_kFn(nB, Lambda, kappa, ktn)
            if kFn is None:
                return None, Rejection.ROOT_FIND_FAILED

            k0n, kFd, kFu, nn = _shell_quantities(kFn, Lambda, kappa)
            eps_arr[i] = _energy_density_above(
                kFn, k0n, kFd, kFu, nn, a, alpha, b, beta)

    if not np.all(np.isfinite(eps_arr)):
        return None, Rejection.NUMERICAL
    if np.any(eps_arr <= 0):
        return None, Rejection.EPS_NONPOSITIVE

    cs2, P_arr, tag = cs2_from_eps_array(eps_arr)
    if cs2 is None:
        return None, Rejection.DE_NONPOSITIVE

    if not np.all(np.isfinite(cs2)):
        return None, Rejection.NUMERICAL
    if np.any(P_arr <= 0):
        return None, Rejection.P_NONPOSITIVE

    return cs2, None



def _draw_params(rng,
                 Lambda_range=None, nt_range=None,
                 a_range=None, alpha_range=None,
                 b_range=None, beta_range=None):
    L_lo, L_hi = Lambda_range or (LAMBDA_MIN, LAMBDA_MAX)
    nt_lo, nt_hi = nt_range or (NT_OVER_N0_MIN, NT_OVER_N0_MAX)
    a_lo, a_hi = a_range or (A_MIN, A_MAX)
    al_lo, al_hi = alpha_range or (ALPHA_MIN, ALPHA_MAX)
    b_lo, b_hi = b_range or (B_MIN, B_MAX)
    be_lo, be_hi = beta_range or (BETA_MIN, BETA_MAX)
    return (uniform(L_lo, L_hi, rng=rng),
            uniform(nt_lo, nt_hi, rng=rng),
            uniform(a_lo, a_hi, rng=rng),
            uniform(al_lo, al_hi, rng=rng),
            uniform(b_lo, b_hi, rng=rng),
            uniform(be_lo, be_hi, rng=rng))


def _s_broad_uniform(rng=None):
    return _draw_params(rng)


def _s_early_transition(rng=None):
    return _draw_params(rng, nt_range=(1.0, 2.5))


def _s_late_transition(rng=None):
    return _draw_params(rng, nt_range=(3.0, NT_OVER_N0_MAX))


def _s_large_lambda(rng=None):
    return _draw_params(rng, Lambda_range=(400.0, LAMBDA_MAX),
                        nt_range=(2.0, NT_OVER_N0_MAX))


def _s_small_lambda(rng=None):
    return _draw_params(rng, Lambda_range=(LAMBDA_MIN, 300.0))


def _s_stiff_repulsion(rng=None):
    return _draw_params(rng, b_range=(15.0, B_MAX),
                        beta_range=(2.5, BETA_MAX))


def _s_soft_interaction(rng=None):
    return _draw_params(rng, b_range=(B_MIN, 10.0))


def _s_mclerran_reddy_like(rng=None):
    a = normal_truncated(-28.6, 5.0, A_MIN, A_MAX, rng=rng)
    alpha = normal_truncated(1.0, 0.2, ALPHA_MIN, ALPHA_MAX, rng=rng)
    b = normal_truncated(9.9, 4.0, B_MIN, B_MAX, rng=rng)
    beta = normal_truncated(2.0, 0.3, BETA_MIN, BETA_MAX, rng=rng)
    Lambda = uniform(LAMBDA_MIN, LAMBDA_MAX, rng=rng)
    nt = uniform(NT_OVER_N0_MIN, NT_OVER_N0_MAX, rng=rng)
    return Lambda, nt, a, alpha, b, beta


def _s_gandolfi_like(rng=None):
    a = normal_truncated(13.0, 5.0, A_MIN, A_MAX, rng=rng)
    alpha = normal_truncated(0.5, 0.15, ALPHA_MIN, ALPHA_MAX, rng=rng)
    b = normal_truncated(3.0, 3.0, B_MIN, B_MAX, rng=rng)
    beta = normal_truncated(2.4, 0.3, BETA_MIN, BETA_MAX, rng=rng)
    Lambda = uniform(LAMBDA_MIN, LAMBDA_MAX, rng=rng)
    nt = uniform(NT_OVER_N0_MIN, NT_OVER_N0_MAX, rng=rng)
    return Lambda, nt, a, alpha, b, beta


def _s_normal_centered(rng=None):
    Lambda = normal_truncated(350.0, 100.0, LAMBDA_MIN, LAMBDA_MAX, rng=rng)
    nt = normal_truncated(2.5, 1.0, NT_OVER_N0_MIN, NT_OVER_N0_MAX, rng=rng)
    a = normal_truncated(-10.0, 15.0, A_MIN, A_MAX, rng=rng)
    alpha = normal_truncated(0.8, 0.3, ALPHA_MIN, ALPHA_MAX, rng=rng)
    b = normal_truncated(10.0, 7.0, B_MIN, B_MAX, rng=rng)
    beta = normal_truncated(2.2, 0.4, BETA_MIN, BETA_MAX, rng=rng)
    return Lambda, nt, a, alpha, b, beta


def _s_beta_centered(rng=None):
    Lambda = beta_scaled(2.0, 2.0, LAMBDA_MIN, LAMBDA_MAX, rng=rng)
    nt = beta_scaled(2.0, 2.0, NT_OVER_N0_MIN, NT_OVER_N0_MAX, rng=rng)
    a = beta_scaled(2.0, 2.0, A_MIN, A_MAX, rng=rng)
    alpha = beta_scaled(2.0, 2.0, ALPHA_MIN, ALPHA_MAX, rng=rng)
    b = beta_scaled(2.0, 2.0, B_MIN, B_MAX, rng=rng)
    beta_val = beta_scaled(2.0, 2.0, BETA_MIN, BETA_MAX, rng=rng)
    return Lambda, nt, a, alpha, b, beta_val


STRATEGIES = [
    _s_broad_uniform, _s_early_transition, _s_late_transition,
    _s_large_lambda, _s_small_lambda,
    _s_stiff_repulsion, _s_soft_interaction,
    _s_mclerran_reddy_like, _s_gandolfi_like,
    _s_normal_centered, _s_beta_centered,
]


def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng
        rng = get_rng()
    strategy = STRATEGIES[rng.integers(len(STRATEGIES))]
    Lambda, nt_over_n0, a, alpha, b, beta = strategy(rng=rng)

    cs2, reason = _evaluate_on_grid(Lambda, nt_over_n0, a, alpha, b, beta)
    if cs2 is None:
        return None, strategy.__name__, reason

    tag = cs2_acceptance_check(cs2)
    if tag is not None:
        return None, strategy.__name__, Rejection[tag]

    return cs2, strategy.__name__, None
