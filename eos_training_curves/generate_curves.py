"""
Generate training data for the EOS diffusion model.

Usage:
    python generate_curves.py --cls 1 --n_samples 60000 --seed 42
    python generate_curves.py --cls 1 --n_samples 1000 --save_pt
    python generate_curves.py --combine class1_polytrope_data.npy class2_spectral_data.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import Counter
class _QuotaTag:
    """Stands in for a Rejection enum member in the rejection Counter."""
    name = "NONPT_QUOTA_FULL"
    def __repr__(self):
        return "NONPT_QUOTA_FULL"

_NONPT_QUOTA_FULL = _QuotaTag()

# MUST stay in sync with diagnostics.phase_transition_diagnostic
def pt_flag(cs2, threshold=0.05, lo=1.5, hi=6.0):
    from eos_common import NB_OVER_N0_GRID as g
    wp = np.where((g >= lo) & (g <= hi))[0]
    dip = wp[np.argmin(cs2[wp])]
    pk = wp[np.argmax(cs2[wp])]
    return bool(cs2[dip] < threshold) and bool(g[dip] > g[pk])

CLASS_REGISTRY = {}

def register_class(cls_id, module_name):
    CLASS_REGISTRY[cls_id] = module_name

register_class(1, 'class1_polytrope')
register_class(2, 'class2_spectral')
register_class(3, 'class3_cs2_interpolation')
register_class(4, 'class4_rmf')
register_class(5, 'class5_css')
register_class(6, 'class6_quarkyonic')
register_class(7, 'class7_njl_crossover')
register_class(8, 'class8_pqcd')
register_class(9, 'class9_pairwise_convex')
register_class(10, 'class10_dirichlet')

def load_class_module(cls_id):
    import importlib
    if cls_id not in CLASS_REGISTRY:
        available = sorted(CLASS_REGISTRY.keys())
        raise ValueError(f"Class {cls_id} not registered. Available: {available}")
    return importlib.import_module(CLASS_REGISTRY[cls_id])

def generate_class(class_module, N_target, seed=42, max_attempts_factor=50,
                   parent_cls=None, pair_idx=None,
                   pt_min_fraction=None, pt_threshold=0.05):
    from sampling_utils import set_seed
    rng = set_seed(seed)

    curves = []
    strategy_counts = Counter()
    rejection_counts = Counter()
    n_attempts = 0
    max_attempts = N_target * max_attempts_factor
    n_pt = 0
    n_nonpt = 0
    nonpt_cap = None
    if pt_min_fraction is not None:
        nonpt_cap = N_target - int(np.ceil(pt_min_fraction * N_target))

    t0 = time.time()
    while len(curves) < N_target and n_attempts < max_attempts:
        n_attempts += 1
        kwargs = {"rng": rng}
        if parent_cls is not None:
            kwargs["parent_cls"] = parent_cls
        if pair_idx is not None:
            kwargs["pair_idx"] = pair_idx
        result, strat_name, reason = class_module.generate_one_sample(**kwargs)

        if result is not None:
            if nonpt_cap is None:
                curves.append(result)
                strategy_counts[strat_name] += 1
            elif pt_flag(result, threshold=pt_threshold):
                curves.append(result)
                strategy_counts[strat_name] += 1
                n_pt += 1
            elif n_nonpt < nonpt_cap:
                curves.append(result)
                strategy_counts[strat_name] += 1
                n_nonpt += 1
            else:
                rejection_counts[_NONPT_QUOTA_FULL] += 1
        else:
            rejection_counts[reason] += 1

        if n_attempts % 5000 == 0:
            elapsed = time.time() - t0
            rate = len(curves) / n_attempts if n_attempts > 0 else 0
            eta = ((N_target - len(curves)) / (len(curves) / elapsed)
                   if len(curves) > 0 else float('inf'))
            print(f"  Attempts: {n_attempts:>8d} | Accepted: {len(curves):>6d} "
                  f"({rate:.1%}) | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    curves = np.array(curves)
    acc_rate = len(curves) / n_attempts if n_attempts > 0 else 0
    print(f"\nDone. {len(curves)} curves accepted from {n_attempts} attempts "
          f"({acc_rate:.1%}) in {elapsed:.1f}s.")
    if len(curves) < N_target:
        print(f"  WARNING: only {len(curves)}/{N_target} reached "
              f"(hit max_attempts={max_attempts}). "
              f"Raise max_attempts_factor and re-run this class.")
    if nonpt_cap is not None and len(curves) > 0:
        print(f"  PT floor: {n_pt}/{len(curves)} flagged "
              f"({n_pt/len(curves):.1%})  --  requested >= "
              f"{pt_min_fraction:.0%} at threshold {pt_threshold}")
        if n_pt < int(np.ceil(pt_min_fraction * len(curves))):
            print("  WARNING: PT floor NOT met (max_attempts exhausted). "
                  "Raise --max_attempts_factor or lower --pt_min_fraction.")

    # Strategy breakdown
    print(f"\n{'Strategy breakdown':=^60}")
    for name, count in strategy_counts.most_common():
        print(f"  {name:35s}: {count:>6d} ({count/len(curves):>5.1%})")

    # Rejection breakdown
    total_rej = sum(rejection_counts.values())
    print(f"\n{'Rejection breakdown':=^60}")
    print(f"  {'Total rejections':35s}: {total_rej:>6d}")
    for reason, count in rejection_counts.most_common():
        print(f"  {reason.name:35s}: {count:>6d} ({count/total_rej:>5.1%})")

    return curves, strategy_counts, rejection_counts, n_attempts

def plot_diagnostics(data, nB_grid, class_name, outpath,
                     strategy_counts=None, rejection_counts=None):
    has_stats = (strategy_counts is not None and rejection_counts is not None)
    nrows = 3 if has_stats else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    fig.suptitle(f"{class_name}  —  {len(data)} samples", fontsize=14, y=0.98)

    ax = axes[0, 0]
    n_show = min(80, len(data))
    for i in range(n_show):
        ax.plot(nB_grid, data[i], alpha=0.25, lw=0.7)
    ax.axhline(1/3, color='gray', ls='--', alpha=0.5, label=r"$c_s^2 = 1/3$")
    ax.set(xlabel=r"$n_B / n_0$", ylabel=r"$c_s^2$",
           xlim=(0.5, 8), ylim=(0, 1), title=f"Sample curves ({n_show})")
    ax.legend()

    ax = axes[0, 1]
    med = np.median(data, axis=0)
    lo16, hi84 = np.percentile(data, 16, axis=0), np.percentile(data, 84, axis=0)
    lo5, hi95 = np.percentile(data, 5, axis=0), np.percentile(data, 95, axis=0)
    ax.fill_between(nB_grid, lo5, hi95, alpha=0.15, color='C0', label="90%")
    ax.fill_between(nB_grid, lo16, hi84, alpha=0.3, color='C0', label="68%")
    ax.plot(nB_grid, med, 'k-', lw=1.5, label="Median")
    ax.axhline(1/3, color='gray', ls='--', alpha=0.5)
    ax.set(xlabel=r"$n_B / n_0$", ylabel=r"$c_s^2$",
           xlim=(0.5, 8), ylim=(0, 1), title="Credible bands")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for nB_val in [1.0, 2.0, 4.0, 6.0]:
        idx = np.argmin(np.abs(nB_grid - nB_val))
        ax.hist(data[:, idx], bins=40, alpha=0.4, density=True,
                label=rf"$n_B/n_0 = {nB_val}$")
    ax.set(xlabel=r"$c_s^2$", ylabel="Density", title="Marginal distributions")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    max_cs2 = np.max(data, axis=1)
    thresholds = np.linspace(0, 1, 200)
    fractions = [np.mean(max_cs2 > t) for t in thresholds]
    ax.plot(thresholds, fractions, 'k-', lw=2)
    ax.axvline(1/3, color='gray', ls='--', alpha=0.5, label=r"$1/3$")
    ax.set(xlabel=r"threshold $c_s^{2*}$",
           ylabel=r"fraction with max$(c_s^2) > c_s^{2*}$",
           title="Survival function of max $c_s^2$",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    if has_stats:
        ax = axes[2, 0]
        names = [n.replace('_s_', '') for n in strategy_counts.keys()]
        counts = list(strategy_counts.values())
        ax.barh(names, counts, color='C0', alpha=0.7)
        ax.set(xlabel="Accepted curves", title="Strategy breakdown")

        ax = axes[2, 1]
        if rejection_counts:
            rnames = [r.name for r in rejection_counts.keys()]
            rcounts = list(rejection_counts.values())
            colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(rcounts)))
            ax.barh(rnames, rcounts, color=colors)
            ax.set(xlabel="Rejected samples", title="Rejection breakdown")
        else:
            ax.text(0.5, 0.5, "No rejections!", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title("Rejection breakdown")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Diagnostic plot saved to {outpath}")


def combine_classes(npy_paths, outdir, normalise=True):
    from eos_common import EOSDataset, NB_OVER_N0_GRID

    arrays = []
    for p in npy_paths:
        arr = np.load(p)
        print(f"  Loaded {p}: {arr.shape}")
        arrays.append(arr)
    combined = np.concatenate(arrays, axis=0)
    print(f"  Combined shape: {combined.shape}")

    rng = np.random.default_rng(0)
    rng.shuffle(combined)

    npy_path = os.path.join(outdir, "combined_data.npy")
    np.save(npy_path, combined)
    print(f"  Saved {npy_path}")

    dataset = EOSDataset(combined, normalise=normalise)
    pt_path = os.path.join(outdir, "combined_data.pt")
    dataset.save(pt_path)
    print(f"  Saved {pt_path}")

    plot_diagnostics(combined, NB_OVER_N0_GRID, "Combined training set",
                     os.path.join(outdir, "combined_diag.png"))
    return dataset



def combine_train_val(train_npy_paths, val_npy_paths, out_path):
    import torch
    from eos_common import NB_OVER_N0_GRID

    train = np.concatenate([np.load(p) for p in train_npy_paths], axis=0)
    np.random.default_rng(0).shuffle(train)
    train_t = torch.from_numpy(train).float()
    mean = train_t.mean(dim=0)
    std  = torch.clamp(train_t.std(dim=0), min=1e-8)
    train_norm = (train_t - mean) / std

    val = np.concatenate([np.load(p) for p in val_npy_paths], axis=0)
    np.random.default_rng(1).shuffle(val)
    val_t = torch.from_numpy(val).float()

    torch.save({
        'train_data':      train_norm,
        'mean':            mean,
        'std':             std,
        'val_data':        val_t,
        'normalise':       True,
        'n_grid':          train_norm.shape[1],
        'nB_over_n0_grid': torch.from_numpy(NB_OVER_N0_GRID).float(),
    }, out_path)
    print(f"Saved {out_path}: train {tuple(train_norm.shape)} normalized, "
          f"val {tuple(val_t.shape)} raw.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate EOS training data for the diffusion model.")
    parser.add_argument("--cls", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--pt_min_fraction", type=float, default=None)
    parser.add_argument("--pt_threshold", type=float, default=0.05)
    parser.add_argument("--max_attempts_factor", type=int, default=50)
    parser.add_argument("--combine", nargs="+", type=str, default=None)
    parser.add_argument("--combine_tv", action="store_true",
                        help="Combine train+val .npy lists into one .pt "
                             "(use with --train_npys, --val_npys, --out_pt)")
    parser.add_argument("--train_npys", nargs="+", type=str, default=None)
    parser.add_argument("--val_npys", nargs="+", type=str, default=None)
    parser.add_argument("--out_pt", type=str, default="eos_train_val.pt")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.combine is not None:
        print(f"Combining {len(args.combine)} files...\n")
        combine_classes(args.combine, args.outdir)
        exit(0)

    if args.combine_tv:
        if not args.train_npys or not args.val_npys:
            parser.error("--combine_tv requires --train_npys and --val_npys")
        combine_train_val(args.train_npys, args.val_npys,
                          os.path.join(args.outdir, args.out_pt))
        exit(0)

    cls_mod = load_class_module(args.cls)
    class_name = cls_mod.CLASS_NAME
    file_prefix = cls_mod.FILE_PREFIX

    from eos_common import EPS_REF, P_REF, NB_OVER_N0_GRID

    print(f"SLy4 reference point at nB/n0 = 0.5:")
    print(f"  ε_ref = {EPS_REF:.3f} MeV/fm³")
    print(f"  P_ref = {P_REF:.4f} MeV/fm³")
    print(f"  P/ε   = {P_REF/EPS_REF:.5f}\n")

    print(f"Generating {args.n_samples} curves for {class_name} "
          f"(seed={args.seed})...\n")
    if args.cls == 5:
        parent_names = ["Class 1 (polytrope)", "Class 2 (spectral)",
                        "Class 3 (cs2 interp)", "Class 4 (RMF)"]
        n_per_parent = args.n_samples // 4
        remainders = args.n_samples - 4 * n_per_parent
        all_curves = []
        strat_counts = Counter()
        rej_counts = Counter()
        total_att = 0

        for pcls in range(4):
            n_this = n_per_parent + (1 if pcls < remainders else 0)
            print(f"\n{'':─^60}")
            print(f"  Parent {pcls}: {parent_names[pcls]}  —  "
                  f"target {n_this} curves")
            print(f"{'':─^60}")
            curves_p, sc, rc, na = generate_class(
                cls_mod, n_this, seed=args.seed + pcls, parent_cls=pcls,
                max_attempts_factor=args.max_attempts_factor,
                pt_min_fraction=args.pt_min_fraction,
                pt_threshold=args.pt_threshold)
            all_curves.append(curves_p)
            strat_counts += sc
            rej_counts += rc
            total_att += na

        data = np.concatenate(all_curves, axis=0)
        shuffle_rng = np.random.default_rng(args.seed)
        shuffle_rng.shuffle(data)

        print(f"\n{'COMBINED CLASS 5 SUMMARY':=^60}")
        print(f"  Total: {len(data)} curves from {total_att} attempts")
        for pcls in range(4):
            print(f"  {parent_names[pcls]:30s}: {len(all_curves[pcls]):>6d}")
    elif args.cls == 9:
        from class9_pairwise_convex import _ALLOWED_PAIRS, _CLASS_LABELS
        n_pairs = len(_ALLOWED_PAIRS)  # 25
        n_per_pair = args.n_samples // n_pairs
        remainders = args.n_samples - n_pairs * n_per_pair
        all_curves = []
        strat_counts = Counter()
        rej_counts = Counter()
        total_att = 0

        for pi, (a, b) in enumerate(_ALLOWED_PAIRS):
            n_this = n_per_pair + (1 if pi < remainders else 0)
            pair_label = (f"{{{a}, {b}}}  "
                          f"({_CLASS_LABELS[a]} × {_CLASS_LABELS[b]})")
            print(f"\n{'':─^60}")
            print(f"  Pair {pi:>2d}/{n_pairs - 1}: {pair_label}  —  "
                  f"target {n_this} curves")
            print(f"{'':─^60}")
            curves_p, sc, rc, na = generate_class(
                cls_mod, n_this, seed=args.seed + pi, pair_idx=pi,
                max_attempts_factor=args.max_attempts_factor,
                pt_min_fraction=args.pt_min_fraction,
                pt_threshold=args.pt_threshold)
            all_curves.append(curves_p)
            strat_counts += sc
            rej_counts += rc
            total_att += na

        data = np.concatenate(all_curves, axis=0)
        shuffle_rng = np.random.default_rng(args.seed)
        shuffle_rng.shuffle(data)

        print(f"\n{'COMBINED CLASS 9 SUMMARY':=^60}")
        print(f"  Total: {len(data)} curves from {total_att} attempts")
        for pi, (a, b) in enumerate(_ALLOWED_PAIRS):
            pair_label = f"{{{a}, {b}}} {_CLASS_LABELS[a]}×{_CLASS_LABELS[b]}"
            print(f"  {pair_label:40s}: {len(all_curves[pi]):>6d}")

    else:
        data, strat_counts, rej_counts, total_att = generate_class(
            cls_mod, args.n_samples, seed=args.seed,
            max_attempts_factor=args.max_attempts_factor,
            pt_min_fraction=args.pt_min_fraction,
            pt_threshold=args.pt_threshold)

    npy_path = os.path.join(args.outdir, f"{file_prefix}_data.npy")
    np.save(npy_path, data)
    print(f"\nData saved to {npy_path}  —  shape {data.shape}")

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
    plot_diagnostics(data, NB_OVER_N0_GRID, class_name, plot_path,
                     strat_counts, rej_counts)
