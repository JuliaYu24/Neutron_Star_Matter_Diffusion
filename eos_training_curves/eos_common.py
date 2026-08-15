"""
Shared infrastructure for all EOS functional classes.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

n0 = 0.16              # nuclear saturation density [fm^-2]
mN = 939.0             # average nucleon mass [MeV]
m0_g = 1.66054e-24     # atomic mass unit [g]
c_cgs = 2.99792458e10  # speed of light [cm/s]
MeV_fm3_to_dyn_cm2 = 1.60218e33  # 1 MeV/fm^3 → dyn/cm^2


hc  = 197.3269804
hc3 = hc**3                # (ℏc)^3 [MeV^3*fm^3]


def rho_to_eps(rho):
    return rho * c_cgs**2 / MeV_fm3_to_dyn_cm2


def P_cgs_to_mev(P_dyn):
    return P_dyn / MeV_fm3_to_dyn_cm2

def fermi_energy_integral(kF, m):
    if kF <= 0:
        return 0.0
    E = np.sqrt(kF * kF + m * m)
    return 0.125 * (kF * E * (2 * kF * kF + m * m)
                    - m**4 * np.log((kF + E) / m))


def fermi_pressure_integral(kF, m):
    if kF <= 0:
        return 0.0
    E = np.sqrt(kF * kF + m * m)
    return (1.0 / 24) * (kF * E * (2 * kF * kF - 3 * m * m)
                         + 3 * m**4 * np.log((kF + E) / m))


def fermi_scalar_integral(kF, m):
    if kF <= 0:
        return 0.0
    E = np.sqrt(kF * kF + m * m)
    return 0.5 * m * (kF * E - m * m * np.log((kF + E) / m))


def kF_from_density(n):
    return (3.0 * np.pi**2 * n) ** (1.0 / 3) * hc if n > 0 else 0.0


def density_from_kF(kF):
    return (kF / hc) ** 3 / (3.0 * np.pi**2)


def cs2_from_eps_array(eps_arr, nB_arr=None):
    if nB_arr is None:
        nB_arr = NB_GRID
    N = len(eps_arr)
    mu = np.empty(N)
    for k in range(N):
        if k == 0:
            mu[k] = (eps_arr[1] - eps_arr[0]) / (nB_arr[1] - nB_arr[0])
        elif k == N - 1:
            mu[k] = (eps_arr[-1] - eps_arr[-2]) / (nB_arr[-1] - nB_arr[-2])
        else:
            mu[k] = ((eps_arr[k + 1] - eps_arr[k - 1])
                      / (nB_arr[k + 1] - nB_arr[k - 1]))
    P_arr = nB_arr * mu - eps_arr
    cs2 = np.empty(N)
    for k in range(N):
        if k == 0:
            dp = P_arr[1] - P_arr[0]
            de = eps_arr[1] - eps_arr[0]
        elif k == N - 1:
            dp = P_arr[-1] - P_arr[-2]
            de = eps_arr[-1] - eps_arr[-2]
        else:
            dp = P_arr[k + 1] - P_arr[k - 1]
            de = eps_arr[k + 1] - eps_arr[k - 1]
        if de <= 0:
            return None, None, "DE_NONPOSITIVE"
        cs2[k] = dp / de

    return cs2, P_arr, None

_a = [None,
      6.22, 6.121, 0.005925, 0.16326,
      6.48, 11.4971, 19.105, 0.8938,
      6.54, 11.4950, -22.775, 1.5707,
      4.3, 14.08, 27.80, -1.653,
      1.50, 14.67]


def _f0(x):
    return 1.0 / (np.exp(np.clip(x, -500, 500)) + 1.0)


def sly4_pressure_cgs(rho):
    a = _a
    xi = np.log10(rho)
    zeta = ((a[1] + a[2]*xi + a[3]*xi**3) / (1 + a[4]*xi)
            * _f0(a[5]*(xi - a[6]))
            + (a[7] + a[8]*xi) * _f0(a[9]*(a[10] - xi))
            + (a[11] + a[12]*xi) * _f0(a[13]*(a[14] - xi))
            + (a[15] + a[16]*xi) * _f0(a[17]*(a[18] - xi)))
    return 10.0**zeta


_rho_s = 7.86
_n_s_cgs = _rho_s / m0_g
_n_s_fm3 = _n_s_cgs * 1e-39

def _build_consistent_sly4():
    log_rho = np.linspace(np.log10(_rho_s), 15.5, 8000)
    rho_arr = 10.0**log_rho
    P_arr = sly4_pressure_cgs(rho_arr)
    integrand = c_cgs**2 / (P_arr + rho_arr * c_cgs**2)

    ln_n_ratio = np.zeros_like(rho_arr)
    for i in range(1, len(rho_arr)):
        drho = rho_arr[i] - rho_arr[i - 1]
        ln_n_ratio[i] = ln_n_ratio[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * drho

    n_arr_cgs = _n_s_cgs * np.exp(ln_n_ratio)
    n_arr_fm3 = n_arr_cgs * 1e-39

    eps_arr = rho_to_eps(rho_arr)
    P_arr_mev = P_cgs_to_mev(P_arr)
    assert np.all(np.diff(n_arr_fm3) > 0), "n(ρ) not monotone"
    
    eps_of_nB = interp1d(n_arr_fm3, eps_arr, kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
    P_of_nB = interp1d(n_arr_fm3, P_arr_mev, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
    return eps_of_nB, P_of_nB, n_arr_fm3, eps_arr, P_arr_mev


_eps_of_nB, _P_of_nB, _n_consistent, _eps_consistent, _P_consistent = \
    _build_consistent_sly4()


def sly4_eos(nB):
    return float(_eps_of_nB(nB)), float(_P_of_nB(nB))

NB_REF = 0.5 * n0
EPS_REF, P_REF = sly4_eos(NB_REF)


N_GRID = 200
NB_OVER_N0_GRID = np.linspace(0.5, 8.0, N_GRID)
NB_GRID = NB_OVER_N0_GRID * n0

_nB_lookup = np.linspace(NB_GRID[0], NB_GRID[-1], 500)
_eps_lookup = np.array([sly4_eos(nB)[0] for nB in _nB_lookup])
nB_to_eps_interp = interp1d(_nB_lookup, _eps_lookup, kind='linear')

EPS_INTEGRATION_MAX = 2500.0
N_ODE_POINTS = 2000


def build_pressure_from_cs2(cs2_func,
                            eps_max=EPS_INTEGRATION_MAX,
                            n_ode=N_ODE_POINTS):
    eps_eval = np.linspace(EPS_REF, eps_max, n_ode)

    def rhs(eps, P):
        return [cs2_func(eps)]

    try:
        sol = solve_ivp(rhs, [EPS_REF, eps_max], [P_REF],
                        t_eval=eps_eval, method='RK45',
                        rtol=1e-8, atol=1e-12)
    except Exception:
        return None

    if not sol.success:
        return None

    P_arr = sol.y[0]

    if np.any(~np.isfinite(P_arr)) or np.any(P_arr <= 0):
        return None

    return interp1d(sol.t, P_arr, kind='linear',
                    bounds_error=False, fill_value='extrapolate')

def evaluate_cs2_on_grid(pressure_func, cs2_func,
                         eps_max=EPS_INTEGRATION_MAX,
                         n_ode=N_ODE_POINTS):
    eps_eval = np.linspace(EPS_REF, eps_max, n_ode)

    def rhs(eps, nB):
        return nB / (eps + pressure_func(eps))

    try:
        sol = solve_ivp(rhs, [EPS_REF, eps_max], [NB_REF],
                        t_eval=eps_eval, method='RK45',
                        rtol=1e-8, atol=1e-12)
    except Exception:
        return None

    if not sol.success:
        return None

    eps_arr, nB_arr = sol.t, sol.y[0]

    if nB_arr[-1] < NB_GRID[-1]:
        return None
    if np.any(np.diff(nB_arr) <= 0):
        return None

    try:
        eps_of_nB = interp1d(nB_arr, eps_arr, kind='linear')
    except Exception:
        return None

    if NB_GRID[0] < nB_arr[0] or NB_GRID[-1] > nB_arr[-1]:
        return None

    eps_on_grid = eps_of_nB(NB_GRID)
    cs2_on_grid = np.array([cs2_func(e) for e in eps_on_grid])

    return cs2_on_grid

P_INTEGRATION_MAX = 3000.0
N_ODE_POINTS_P = 2000


def evaluate_cs2_on_grid_P(Gamma_func,
                           P_max=P_INTEGRATION_MAX,
                           n_ode=N_ODE_POINTS_P):
    P_eval = np.linspace(P_REF, P_max, n_ode)

    def rhs(P, y):
        eps, nB = y
        G = Gamma_func(P)
        if G <= 0 or not np.isfinite(G):
            return [0.0, 0.0]
        GP = G * P
        return [(eps + P) / GP, nB / GP]

    try:
        sol = solve_ivp(rhs, [P_REF, P_max], [EPS_REF, NB_REF],
                        t_eval=P_eval, method='RK45',
                        rtol=1e-8, atol=1e-12)
    except Exception:
        return None

    if not sol.success:
        return None

    P_arr = sol.t
    eps_arr, nB_arr = sol.y[0], sol.y[1]

    if nB_arr[-1] < NB_GRID[-1]:
        return None
    if np.any(np.diff(nB_arr) <= 0):
        return None

    try:
        P_of_nB = interp1d(nB_arr, P_arr, kind='linear')
        eps_of_nB = interp1d(nB_arr, eps_arr, kind='linear')
    except Exception:
        return None

    if NB_GRID[0] < nB_arr[0] or NB_GRID[-1] > nB_arr[-1]:
        return None

    P_on_grid = P_of_nB(NB_GRID)
    eps_on_grid = eps_of_nB(NB_GRID)
    Gamma_on_grid = np.array([Gamma_func(p) for p in P_on_grid])
    cs2_on_grid = Gamma_on_grid * P_on_grid / (eps_on_grid + P_on_grid)

    return cs2_on_grid
def cs2_acceptance_check(cs2, min_std=0.01):
    if np.any(~np.isfinite(cs2)):
        return "NUMERICAL"
    if np.any(cs2 > 1.0):
        return "ACAUSAL"
    if np.any(cs2 < 0.0):
        return "UNSTABLE"
    if np.std(cs2) < min_std:
        return "LOW_VARIATION"
    return None

def thermodynamic_check(cs2_on_grid, max_P_over_eps=0.85,
                        min_eps_per_baryon=900.0,
                        max_eps_per_baryon=4000.0):
    nB = NB_GRID
    dnB = np.diff(nB)
    N = len(nB)

    eps = np.empty(N)
    P = np.empty(N)
    eps[0] = EPS_REF
    P[0] = P_REF
    for i in range(N - 1):
        ep = eps[i] + P[i]
        if ep <= 0:
            return "EPS_PLUS_P_NONPOSITIVE", None, None
        deps = ep / nB[i] * dnB[i]
        dP = cs2_on_grid[i] * ep / nB[i] * dnB[i]
        eps[i + 1] = eps[i] + deps
        P[i + 1] = P[i] + dP
    if np.any(P <= 0):
        return "P_NONPOSITIVE", None, None
    if np.any(eps <= 0):
        return "EPS_NONPOSITIVE", None, None
    if np.any(np.diff(P) < 0):
        return "P_NON_MONOTONE", None, None
    ratio = P / eps
    if np.any(ratio > max_P_over_eps):
        return "P_OVER_EPS_TOO_LARGE", None, None
    eps_per_nB = eps / nB
    if np.any(eps_per_nB < min_eps_per_baryon):
        return "EPS_PER_BARYON_TOO_LOW", None, None
    if eps_per_nB[-1] > max_eps_per_baryon:
        return "EPS_PER_BARYON_TOO_HIGH", None, None
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(P))):
        return "NUMERICAL", None, None

    return None, eps, P

try:
    import torch
    from torch.utils.data import Dataset

    class EOSDataset(Dataset):
        def __init__(self, data, normalise=True, mean=None, std=None):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)

            self.raw_data = data
            self.normalise = normalise
            if mean is not None and std is not None:
                self.mean = mean.clone() if torch.is_tensor(mean) else torch.tensor(mean).float()
                self.std  = std.clone()  if torch.is_tensor(std)  else torch.tensor(std).float()
            else:
                self.mean = data.mean(dim=0)
                self.std  = data.std(dim=0)
            self.std = torch.clamp(self.std, min=1e-8)
            if normalise:
                self.data = (data - self.mean) / self.std
            else:
                self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def unnormalise(self, x):
            return x * self.std + self.mean

        def get_normalisation(self):
            return self.mean.clone(), self.std.clone()

        def save(self, path):
            torch.save({
                'raw_data': self.raw_data,
                'data': self.data,
                'mean': self.mean,
                'std': self.std,
                'normalise': self.normalise,
                'n_grid': self.data.shape[1],
                'nB_over_n0_grid': torch.from_numpy(NB_OVER_N0_GRID).float(),
            }, path)

        @classmethod
        def load(cls, path, normalise=True):
            """Load dataset from a .pt file."""
            checkpoint = torch.load(path, weights_only=False)
            dataset = cls(checkpoint['raw_data'], normalise=normalise)
            return dataset

        @classmethod
        def from_npy(cls, path, normalise=True):
            """Create dataset from a .npy file."""
            data = np.load(path)
            return cls(data, normalise=normalise)

        @classmethod
        def from_multiple(cls, paths, normalise=True):
            arrays = [np.load(p) for p in paths]
            combined = np.concatenate(arrays, axis=0)
            return cls(combined, normalise=normalise)

    PYTORCH_AVAILABLE = True

except ImportError:
    PYTORCH_AVAILABLE = False

    class EOSDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for EOSDataset. "
                "Install with: pip install torch"
            )