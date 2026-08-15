#!/usr/bin/env python3
"""
Measure the natural phase-transition-flag rate of every class .npy file,
and the generation-cost multiplier a PT floor would impose.

The flag is IDENTICAL to generate_curves.pt_flag and to
diagnostics.phase_transition_diagnostic: in the window
n_B/n0 in [1.5, 6.0], the c_s^2 minimum lies below `threshold` AND at
higher density than the in-window maximum.

Run from HPC/eos_training_curves (so eos_common resolves):

    python measure_pt_rates.py data/train
    python measure_pt_rates.py data/val
    python measure_pt_rates.py data/train --thresholds 0.05 0.10 --floor 0.30

Reading the output:
  p(<t)              natural flagged fraction at threshold t
  cost x             extra accepted-draw factor to reach the floor at the
                     FIRST listed threshold ( = floor / p ; 1.0 means the
                     floor is inactive because the class is already above it)
  verdict            'natural' (floor inactive), the cost factor, or
                     'INFEASIBLE' (rate ~ 0: rejection cannot create shapes
                     the generator never produces)
"""

import argparse
import glob
import os
import sys

import numpy as np

from eos_common import NB_OVER_N0_GRID as G


def pt_mask(arr, threshold, lo=1.5, hi=6.0):
    """Vectorized flag over an (N, 200) array; mirrors pt_flag exactly."""
    wp = np.where((G >= lo) & (G <= hi))[0]
    sub = arr[:, wp]
    dip = wp[np.argmin(sub, axis=1)]
    pk = wp[np.argmax(sub, axis=1)]
    return (arr[np.arange(len(arr)), dip] < threshold) & (G[dip] > G[pk])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="data tree split, e.g. data/train")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.05, 0.10])
    ap.add_argument("--floor", type=float, default=0.30)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*", "*_data.npy")))
    if not files:
        sys.exit(f"no */*_data.npy under {args.root}")

    t0 = args.thresholds[0]
    head = f"{'class file':42s}" + "".join(
        f"  p(<{t:g})" for t in args.thresholds)
    head += f"    floor {args.floor:.0%} @ t={t0:g}"
    print(head)
    print("-" * len(head))

    for f in files:
        arr = np.load(f)
        ps = [float(pt_mask(arr, t).mean()) for t in args.thresholds]
        p0 = ps[0]
        if p0 >= args.floor:
            verdict = "natural (floor inactive)"
        elif p0 >= 0.005:
            verdict = f"cost x{args.floor / p0:5.1f}"
        elif p0 > 0.0:
            verdict = f"cost x{args.floor / p0:7.0f}  (impractical)"
        else:
            verdict = "INFEASIBLE (rate ~ 0)"
        print(f"{os.path.basename(f):42s}"
              + "".join(f"  {p:7.3%}" for p in ps)
              + f"    {verdict}")


if __name__ == "__main__":
    main()
