#!/usr/bin/env python3
"""
Driver for the VALIDATION families (classes 11-13), living in
HPC/eos_class_validation_curves/ and reusing the machinery of
HPC/eos_training_curves/generate_curves.py (generation loop, diagnostics
plot, combine utilities) plus eos_common / sampling_utils -- nothing is
duplicated, so the two directories cannot drift.

Usage (mirrors the original driver):

    python generate_validation_curves.py --cls 11 --n_samples 60000 --seed 42
    python generate_validation_curves.py --cls 12 --n_samples 60000 --seed 42
    python generate_validation_curves.py --cls 13 --n_samples 60000 --seed 42

    # combine the 10 training + 3 validation .npy files into one set:
    python generate_validation_curves.py --combine <path1.npy> <path2.npy> ...

    # build the normalized train/val .pt for retraining:
    python generate_validation_curves.py --combine_tv \\
        --train_npys <13 train npys...> --val_npys <13 val npys...> \\
        --out_pt eos13_train_val.pt

Classes:
    11  Gaussian-process sound speed        (class11_gp_cs2.py)
    12  piecewise-linear sound speed        (class12_pwlinear_cs2.py)
    13  nuclear empirical meta-model        (class13_metamodel.py)
Classes 1-10 are generated, as before, with
eos_training_curves/generate_curves.py.
"""

import os
import sys
import argparse

os.environ.setdefault("MPLBACKEND", "Agg")       # headless HPC nodes

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.abspath(os.path.join(_HERE, os.pardir,
                                          "eos_training_curves"))
for _p in (_TRAIN_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import generate_curves as gc                      # original machinery
from eos_common import EPS_REF, P_REF, NB_OVER_N0_GRID

VALIDATION_REGISTRY = {
    11: "class11_gp_cs2",
    12: "class12_pwlinear_cs2",
    13: "class13_metamodel",
}


def load_validation_module(cls_id):
    import importlib
    if cls_id in VALIDATION_REGISTRY:
        return importlib.import_module(VALIDATION_REGISTRY[cls_id])
    if cls_id in gc.CLASS_REGISTRY:
        raise SystemExit(
            f"Class {cls_id} is a TRAINING class -- generate it with "
            f"eos_training_curves/generate_curves.py (this driver only "
            f"handles validation classes {sorted(VALIDATION_REGISTRY)}).")
    raise ValueError(
        f"Class {cls_id} not registered. "
        f"Validation classes: {sorted(VALIDATION_REGISTRY)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate EOS validation-class data (classes 11-13).")
    parser.add_argument("--cls", type=int, default=11)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--max_attempts_factor", type=int, default=50)
    parser.add_argument("--combine", nargs="+", type=str, default=None)
    parser.add_argument("--combine_tv", action="store_true",
                        help="Combine train+val .npy lists into one .pt "
                             "(use with --train_npys, --val_npys, --out_pt)")
    parser.add_argument("--train_npys", nargs="+", type=str, default=None)
    parser.add_argument("--val_npys", nargs="+", type=str, default=None)
    parser.add_argument("--out_pt", type=str, default="eos13_train_val.pt")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.combine is not None:
        print(f"Combining {len(args.combine)} files...\n")
        gc.combine_classes(args.combine, args.outdir)
        sys.exit(0)

    if args.combine_tv:
        if not args.train_npys or not args.val_npys:
            parser.error("--combine_tv requires --train_npys and --val_npys")
        gc.combine_train_val(args.train_npys, args.val_npys,
                             os.path.join(args.outdir, args.out_pt))
        sys.exit(0)

    cls_mod = load_validation_module(args.cls)
    class_name = cls_mod.CLASS_NAME
    file_prefix = cls_mod.FILE_PREFIX

    print("SLy4 reference point at nB/n0 = 0.5:")
    print(f"  eps_ref = {EPS_REF:.3f} MeV/fm^3")
    print(f"  P_ref   = {P_REF:.4f} MeV/fm^3")
    print(f"  P/eps   = {P_REF / EPS_REF:.5f}\n")

    print(f"Generating {args.n_samples} curves for {class_name} "
          f"(seed={args.seed})...\n")

    data, strat_counts, rej_counts, total_att = gc.generate_class(
        cls_mod, args.n_samples, seed=args.seed,
        max_attempts_factor=args.max_attempts_factor)

    npy_path = os.path.join(args.outdir, f"{file_prefix}_data.npy")
    np.save(npy_path, data)
    print(f"\nData saved to {npy_path}  --  shape {data.shape}")

    if args.save_pt:
        from eos_common import EOSDataset, PYTORCH_AVAILABLE
        if not PYTORCH_AVAILABLE:
            print("WARNING: PyTorch not available, skipping .pt save.")
        else:
            dataset = EOSDataset(data, normalise=True)
            pt_path = os.path.join(args.outdir, f"{file_prefix}_data.pt")
            dataset.save(pt_path)
            mean, std = dataset.get_normalisation()
            print(f"PyTorch dataset saved to {pt_path}")
            print(f"  mean range [{mean.min():.4f}, {mean.max():.4f}], "
                  f"std range [{std.min():.4f}, {std.max():.4f}]")

    plot_path = os.path.join(args.outdir, f"{file_prefix}_diag.png")
    gc.plot_diagnostics(data, NB_OVER_N0_GRID, class_name, plot_path,
                        strat_counts, rej_counts)
