#!/usr/bin/env python3
"""
extract_anchor_1n0.py  --  companion to cs2_betaeq_anchors.py  (read-only user of it)

Prints the beta-equilibrium thermodynamic anchor (mu_B, eps, P) at 1.0 n_sat and
neighbouring densities, on the SAME chi-EFT + beta-equilibrium curve that already
defines the 0.5 n_sat reference point of your pipeline.  Use the value at 1.0 n0
(or at the first in-range target-grid point) as the integration constant when the
Annala C4 interpolation ensemble is reconstructed from its own grid start (1.0 n0)
instead of from 0.5 n0.

Why this is the right number
----------------------------
In cs2_betaeq_anchors.py the pressure and energy density come straight from
betaeq_thermo(..., full=True):

    eps = n * e_N + eps_lepton          (e_N includes the rest mass)
    P   = n * mu_B - eps                 (Euler relation)

These are ABSOLUTE quantities -- there is no integration constant and no lower
anchor involved -- so eps and P at 1.0 n0 are defined just as directly as at
0.5 n0, and lie on exactly the same curve.  Your existing 0.5 reference is
simply refpoint["mean"] = [mu_B, eps, P] evaluated at n_ref = 0.08 fm^-3.  This
script evaluates the identical joint draw (same seed, same 'clean' mask, same
reduction) at 1.0 n0 as well, so the anchor is guaranteed
consistent with the 0.5 reference and with 'this work'.

Self-check: the script reprints its own 0.5 n0 value and, if it can find your
stored cs2_BETAEQ_Lambda-<L>_refpoint_mean.npy, compares against it.  If the
0.5 row matches your stored eps_ref/P_ref, the 1.0 row is trustworthy too.
"""

import os
import sys
import argparse
import numpy as np

# --------------------------------------------------------------------------
# Import the user's extraction module (read-only).  Importing it runs its
# legacy-stack compatibility shim and pulls in the nuclear_matter package
# (with a sys.path fallback to the module's own folder), exactly as when the
# original CLI runs.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import cs2_betaeq_anchors as A
except Exception as exc:  # noqa: BLE001
    sys.exit(
        "ERROR: could not import cs2_betaeq_anchors.\n"
        "Put extract_anchor_1n0.py in the SAME folder as cs2_betaeq_anchors.py\n"
        "(that folder must also contain nuclear_matter/ and "
        "all_matter_data_high_density.csv).\n"
        f"underlying error: {exc!r}"
    )

N0 = A.N_SAT  # 0.16 fm^-3


# --------------------------------------------------------------------------
def mean_thermo_curves(csv, Lambda, num_samp, seed):
    """Reproduce the JOINT draw of extract_betaeq_anchor_cov and return the
    sample-mean (mu_B, eps, P) curves on the full density grid.

    Every line here is copied from cs2_betaeq_anchors.extract_betaeq_anchor_cov
    (the joint branch with full=True) and uses the SAME 'clean' mask that its
    reference-point block uses, so the returned curves reproduce
    refpoint['mean'] at any density -- including your stored 0.5 n0 value.
    """
    gp = A.build_joint_gp(csv, Lambda=Lambda)
    n = gp["density_all"]
    N = len(n)
    S_nn, S_ss, C = gp["S_nn"], gp["S_ss"], gp["C"]
    mu_n, mu_s = gp["mu_n"], gp["mu_s"]

    # ---- joint draw, identical to extract_betaeq_anchor_cov ----------------
    rng = np.random.default_rng(seed)
    L_joint = A._symm_psd_sqrt(np.block([[S_nn, C], [C, S_ss]]))
    g = rng.standard_normal((2 * N, num_samp))
    mu_stack = np.concatenate([mu_n, mu_s])[:, None]

    s = L_joint @ g
    E_N = (mu_stack[:N] + s[:N]).T          # (num_samp, N)
    E_S = (mu_stack[N:] + s[N:]).T
    S2 = E_N - E_S
    S2 = np.where(S2 <= 1e-6, 1e-6, S2)     # guard rare non-physical draws
    dE_S = np.gradient(E_S, n, axis=-1)
    dS2 = np.gradient(S2, n, axis=-1)
    t = A.betaeq_thermo(n, E_S, dE_S, S2, dS2, full=True)

    # ---- same 'clean' mask as the reference-point code ---------------------
    # good = samples whose cs2 is finite at all five conditioning anchors.
    anchors = (0.5, 0.75, 1.0, 1.25, 1.5)
    aidx = [int(np.argmin(np.abs(n - a * N0))) for a in anchors]
    good = ~np.any(np.isnan(t["cs2"][:, aidx]), axis=1)

    eps = t["eps"][good].mean(0)
    P = t["P"][good].mean(0)
    muB = t["mu_B"][good].mean(0)
    return n, muB, eps, P, int(good.sum())


def eval_at(nq_fm3, n, muB, eps, P):
    """(mu_B, eps, P) at density nq_fm3 [fm^-3] by linear interp of the mean
    curves.  On grid points (e.g. 0.08, 0.16 fm^-3) this is exact; the only
    interpolated target is the off-grid first-in-range point, where the mean
    curves are smooth and the interpolation error is << 0.1%.
    """
    return (float(np.interp(nq_fm3, n, muB)),
            float(np.interp(nq_fm3, n, eps)),
            float(np.interp(nq_fm3, n, P)))


def find_stored_refpoint(Lambda, explicit=None):
    """Locate the pipeline's stored refpoint_mean .npy for the 0.5 self-check."""
    name = f"cs2_BETAEQ_Lambda-{Lambda}_refpoint_mean.npy"
    cands = [
        explicit or "",
        os.path.join(_HERE, name),
        os.path.join(_HERE, "chEFT", name),
        os.path.join(os.getcwd(), name),
        os.path.join(os.getcwd(), "analysis", "chEFT", name),
        os.path.join(_HERE, "analysis", "chEFT", name),
    ]
    return next((p for p in cands if p and os.path.isfile(p)), None)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Extract the beta-eq (mu_B, eps, P) anchor at 1.0 n_sat.")
    ap.add_argument("--csv", default=None,
                    help="data table (default: found next to cs2_betaeq_anchors.py)")
    ap.add_argument("--Lambda", type=int, default=500, choices=(450, 500))
    ap.add_argument("--num-samp", type=int, default=20000,
                    help="keep the pipeline default (20000) to reproduce your "
                         "stored 0.5 reference exactly")
    ap.add_argument("--seed", type=int, default=12345,
                    help="keep the pipeline default (12345) for the same reason")
    ap.add_argument("--nmin", type=float, default=0.5,
                    help="lower edge of the notebook nB_grid, in n0 (default 0.5)")
    ap.add_argument("--nmax", type=float, default=8.0,
                    help="upper edge of the notebook nB_grid, in n0 (default 8.0)")
    ap.add_argument("--npts", type=int, default=200,
                    help="number of points in the notebook nB_grid (default 200)")
    ap.add_argument("--refpoint-file", default=None,
                    help="explicit path to cs2_BETAEQ_Lambda-<L>_refpoint_mean.npy")
    args = ap.parse_args()

    csv = A.find_data_csv(args.csv)
    print(f"using data: {csv}")
    print(f"Lambda = {args.Lambda} MeV, num_samp = {args.num_samp}, "
          f"seed = {args.seed}\n")

    n, muB, eps, P, ng = mean_thermo_curves(csv, args.Lambda, args.num_samp, args.seed)
    print(f"clean joint samples used: {ng}\n")

    # first in-range point of the notebook's target grid (in n0 and fm^-3):
    tgt_n0 = np.linspace(args.nmin, args.nmax, args.npts)   # n/n0 units
    in_range = tgt_n0 >= 1.0
    first_in_n0 = float(tgt_n0[in_range][0])
    n_oor = int((~in_range).sum())

    # densities to report: 0.5 (self-check), the two 1.0-ish anchors, context
    rows = [
        (0.50, "0.5 n0  -> self-check vs your stored refpoint"),
        (0.75, "0.75 n0"),
        (1.00, "1.0 n0  = Annala C4 native grid start  [ANCHOR option A]"),
        (first_in_n0, f"first in-range target pt        [ANCHOR option B]"),
        (1.25, "1.25 n0"),
        (1.50, "1.50 n0"),
    ]

    print("beta-equilibrium anchors on the mean curve "
          "(same curve as your 0.5 n0 reference):")
    print(f"target grid = linspace({args.nmin:g}, {args.nmax:g}, {args.npts}) n0  "
          f"-> {n_oor} point(s) below Annala's 1.0 n0 start; "
          f"first in-range at {first_in_n0:.4f} n0\n")
    hdr = f"{'n/n0':>8} {'n[fm^-3]':>10} {'mu_B[MeV]':>11} " \
          f"{'eps[MeV/fm^3]':>14} {'P[MeV/fm^3]':>13}   note"
    print(hdr)
    print("-" * len(hdr))
    picks = {}
    for a_n0, note in rows:
        nq = a_n0 * N0
        mu_q, eps_q, P_q = eval_at(nq, n, muB, eps, P)
        picks[round(a_n0, 6)] = (nq, mu_q, eps_q, P_q)
        print(f"{a_n0:8.4f} {nq:10.5f} {mu_q:11.3f} {eps_q:14.4f} {P_q:13.4f}   {note}")

    # ---- self-check against the stored 0.5 reference ----------------------
    print()
    rp_path = find_stored_refpoint(args.Lambda, args.refpoint_file)
    _, _, eps05, P05 = picks[0.5]
    if rp_path is not None:
        stored = np.load(rp_path)  # (3,) = [mu_B, eps, P]
        print(f"self-check vs {rp_path}")
        print(f"  stored  : eps_ref = {stored[1]:.4f}, P_ref = {stored[2]:.4f}")
        print(f"  this run: eps_ref = {eps05:.4f}, P_ref = {P05:.4f}")
        deps = eps05 - stored[1]
        dP = P05 - stored[2]
        print(f"  diff    : d(eps) = {deps:+.4f} ({100*deps/stored[1]:+.3f}%), "
              f"d(P) = {dP:+.4f} ({100*dP/max(abs(stored[2]),1e-9):+.3f}%)")
        if abs(deps) < 1e-2 and abs(dP) < 1e-3:
            print("  --> MATCH: the 1.0 n0 anchor below is trustworthy.\n")
        else:
            print("  --> MISMATCH: check --num-samp/--seed/--Lambda match how the "
                  "stored refpoint was generated before using the 1.0 anchor.\n")
    else:
        print("self-check: stored refpoint_mean.npy not found next to this script.")
        print(f"  this run gives eps_ref(0.5 n0) = {eps05:.4f}, "
              f"P_ref(0.5 n0) = {P05:.4f}")
        print("  (your pipeline value should be ~75.86 / ~0.42; if it is, the "
              "1.0 anchor below is trustworthy.)\n")

    _, muA, epsA, PA = picks[1.0]
    nB_B, muB_B, epsB, PB = picks[round(first_in_n0, 6)]
    print("=" * 72)
    print("Lambda = %d, npe-mu beta-equilibrium, N3LO" % args.Lambda)
    print("=" * 72)
    print("# Option A -- if you integrate C4 from its native grid start, 1.0 n0:")
    print(f"eps_ref_C4 = {epsA:.6f}   # MeV/fm^3  at 1.0 n0 (= {1.0*N0:.4f} fm^-3)")
    print(f"P_ref_C4   = {PA:.6f}     # MeV/fm^3  at 1.0 n0")
    print(f"mu_B_C4    = {muA:.6f}    # MeV       at 1.0 n0  (cross-check only)")
    print()
    print("# Option B -- if you integrate C4 over the in-range TARGET subgrid,")
    print(f"#             whose first point is {first_in_n0:.4f} n0 "
          f"(= {nB_B:.5f} fm^-3):")
    print(f"eps_ref_C4 = {epsB:.6f}   # MeV/fm^3  at {first_in_n0:.4f} n0")
    print(f"P_ref_C4   = {PB:.6f}     # MeV/fm^3  at {first_in_n0:.4f} n0")
    print(f"mu_B_C4    = {muB_B:.6f}    # MeV       at {first_in_n0:.4f} n0")
    print("=" * 72)


if __name__ == "__main__":
    main()