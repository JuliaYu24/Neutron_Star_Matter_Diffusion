#!/usr/bin/env python3
"""
EOS sampling driver: pure DDPM + SciPy importance reweighting.

No sampling-time guidance.  Astrophysical conditioning is applied
entirely as a post-hoc reweighting pass.  This script is a pure
COMPUTATION driver: it runs sampling, reweighting, and writes a .pt
file. 

Edit the configuration blocks below and run:
    python run_sampling.py

Five blocks to edit:
    1. KNOWN DATA POINTS   -- chi-EFT anchors (inpainting).
    2. ANCHOR UNCERTAINTY  -- either independent sigmas OR a full
                              covariance matrix.
    3. ASTRO CONFIG        -- NICER / GW / M_max data + TOV numerics
                              (used by the SciPy reweighter only).
    4. REWEIGHTING CONFIG  -- SciPy tolerances, parallelism.
    5. PQCD CONFIG         -- Komoltsev+2024 marginalized pQCD
                              likelihood.  Set to None to disable.

Any of the astrophysical conditioning terms can be individually
disabled by removing/emptying them in ASTRO_CONFIG:
    * NICER off   : set "nicer_pulsars" to []
    * GW    off   : set "gw"   to None
    * M_max off   : set "mmax" to None
    * pQCD  off   : set PQCD_CONFIG = None
"""

import numpy as np
import torch
from eos_sampling.pipeline import run_sampling
from eos_sampling.astro_configs import NICER_PULSARS_SUMMARY

ANCHOR_DIR = "analysis/chEFT"
LAMBDA     = 500
N_SAT      = 0.16


def _anchor(tag):
    return f"{ANCHOR_DIR}/cs2_BETAEQ_Lambda-{LAMBDA}_anchor_{tag}_5pt.npy"


_nB_anchor_fm3 = np.load(_anchor("nB"))
_nB_anchor_n0  = _nB_anchor_fm3 / N_SAT

nB_known  = torch.from_numpy(_nB_anchor_n0).float()
cs2_known = torch.from_numpy(np.load(_anchor("mean"))).float()
cs2_cov   = torch.from_numpy(np.load(_anchor("cov"))).float()
cs2_sigma = None   # covariance path active

try:
    from eos_common import NB_OVER_N0_GRID as _GRID
except Exception:
    _GRID = np.linspace(0.5, 8.0, 200)

_grid_idx = np.abs(_GRID[None, :] - _nB_anchor_n0[:, None]).argmin(axis=1)
_off_grid = np.abs(_GRID[_grid_idx] - _nB_anchor_n0)
if _off_grid.max() > 1e-9:
    raise SystemExit(
        "chi-EFT anchors are NOT on the training grid:\n"
        + "\n".join(f"    {a:.6f} n0 -> nearest grid point {b:.6f} "
                     f"(off by {c:.2e})"
                     for a, b, c in zip(_nB_anchor_n0, _GRID[_grid_idx],
                                        _off_grid))
        + f"\n  Re-run cs2_betaeq_anchors.py and copy the new .npy files "
          f"into {ANCHOR_DIR}/")

try:
    _idx_file = np.load(_anchor("gridindex"))
except FileNotFoundError:
    _idx_file = None
else:
    if not np.array_equal(_idx_file, _grid_idx):
        raise SystemExit(
            f"grid-index mismatch: extraction recorded {_idx_file.tolist()}, "
            f"this file computes {_grid_idx.tolist()}")

print(f"chEFT anchors  [Lambda = {LAMBDA} MeV, {ANCHOR_DIR}]")
print(f"  {'n/n0':>10} {'nB [fm^-3]':>12} {'component':>10} {'cs2':>10}")
for _u, _nb, _j, _c in zip(_nB_anchor_n0, _nB_anchor_fm3, _grid_idx,
                           cs2_known.numpy()):
    print(f"  {_u:10.6f} {_nb:12.6f} {int(_j):10d} {_c:10.6f}")

def _refpt(tag):
    return f"{ANCHOR_DIR}/cs2_BETAEQ_Lambda-{LAMBDA}_refpoint_{tag}.npy"


_ref_mean  = np.load(_refpt("mean"))   # mu_B, eps, P
_ref_sigma = np.load(_refpt("sigma"))
_ref_nB    = np.load(_refpt("nB"))

EPS_REF_CHEFT = float(_ref_mean[1])
P_REF_CHEFT   = float(_ref_mean[2])
SIGMA_P_REF   = float(_ref_sigma[2])
REF_FILE      = _refpt("mean")

assert abs(float(_ref_nB[0]) - float(_nB_anchor_fm3[0])) < 1e-9, (
    f"reference-point density {float(_ref_nB[0]):.6f} fm^-3 does not match "
    f"the first anchor {float(_nB_anchor_fm3[0]):.6f} fm^-3; the refpoint "
    f"and anchor .npy files are from different extraction runs")
print(f"chEFT reference point  [{REF_FILE}]")
print(f"  eps_ref = {EPS_REF_CHEFT:.4f} MeV/fm^3")
print(f"  P_ref   = {P_REF_CHEFT:.4f} MeV/fm^3   (sigma = {SIGMA_P_REF:.4f})")

ASTRO_CONFIG = {
    "nicer_pulsars": NICER_PULSARS_SUMMARY,

    "nicer_mode": "line_integral",

    "gw": {
        "m1":               1.48,
        "m2":               1.26,
        "Lambda_tilde_obs": 300.0,
        "sigma_plus":       420.0,
        "sigma_minus":      230.0,
        "credible_level":   0.90},
    "mmax": {
        "M_lower_bound": 1.908,
        "sigma":         0.016},

    "n_central":     40,
    "P_c_min":       3.0,
    "P_c_max":       2000.0,
    "r_max":         25.0,

    "eps_ref":       EPS_REF_CHEFT,
    "P_ref":         P_REF_CHEFT,
    "ref_source":    REF_FILE,
}

REWEIGHTING_CONFIG = {
    "nicer_mode": "line_integral",
    "rtol":       1e-6,
    "atol":       1e-8,
    "verbose":    True,
    "n_jobs":     -1,
}

PQCD_CONFIG = {
    "data_path":     "external/zenodo_15407795/"
                     "eos_extensions_s-G-1p25-0p25_l-U-1-20_"
                     "meancs2-G-0.3-0.3_pQCD-25-40.h5",
    "n_T_over_n0":   "n_TOV",
}

SAVE_PREFIX = f"analysis/res_finale_baseline/eos_EFT_posterior_L{LAMBDA}"


if __name__ == "__main__":

    out = run_sampling(
        nB_known           = nB_known,
        cs2_known          = cs2_known,
        cs2_sigma          = cs2_sigma,
        cs2_cov            = cs2_cov,
        checkpoint_path    = "checkpoints/eos_ddpm_best.pt",
        n_samples          = 100000,
        batch_size         = 500,
        seed               = 43,
        enforce_causality  = True,
        cs2_bounds         = (0.0, 1.0),
        astro_config       = ASTRO_CONFIG,
        reweighting_config = REWEIGHTING_CONFIG,
        pqcd_config        = PQCD_CONFIG,     # None disables pQCD
        reuse_raw          = False,
        save_prefix        = SAVE_PREFIX)
    print("\nReturned keys:", list(out.keys()))
    r  = out["reweighting"]
    Nd = out["n_drawn"]
    Np = out["n_post_filter"]
    print(f"\nJoint ESS = {r['ESS']:.1f}")
    print(f"   / N_post_filter = {Np}  ({100.0 * r['ESS'] / Np:.1f}%)")
    print(f"   / N_drawn       = {Nd}  ({100.0 * r['ESS'] / Nd:.1f}%)")

    print(f"\nPer-term ESS (each data term alone):")
    print(f"   NICER  = {r['ESS_nicer']:6.1f}")
    print(f"   GW     = {r['ESS_gw']:6.1f}")
    print(f"   M_max  = {r['ESS_mmax']:6.1f}")

    terms_for_binding = [("NICER", r['ESS_nicer']),
                         ("GW",    r['ESS_gw']),
                         ("M_max", r['ESS_mmax'])]
    if "ESS_pqcd" in r:
        print(f"   pQCD   = {r['ESS_pqcd']:6.1f}")
        terms_for_binding.append(("pQCD", r['ESS_pqcd']))
        n_pass = int(r["pqcd_passed"].sum())
        print(f"   (pQCD: {n_pass}/{r['N']} samples got finite log-L "
              f"under Komoltsev+2024 marginalized window "
              f"= {100.0 * n_pass / r['N']:.1f}%)")

    binder = min(terms_for_binding, key=lambda t: t[1])
    print(f"   -> binding constraint: {binder[0]} (ESS = {binder[1]:.1f})")
    if r["clamp_counts"]:
        total = sum(r["clamp_counts"].values())
        if total > 0:
            print(f"\nPulsar clamp summary (samples hitting -1e6 floor):")
            for pname, cnt in r["clamp_counts"].items():
                pct = 100.0 * cnt / r["N"]
                flag = "  <-- effectively uninformative" if pct > 50 else ""
                print(f"   {pname}: {cnt}/{r['N']}  ({pct:.1f}%){flag}")