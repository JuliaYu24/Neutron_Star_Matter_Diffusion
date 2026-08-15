"""
De-normalization, causality filtering, and summary statistics.
"""

import torch


def denormalize_and_summarize(samples_norm, norm_mean, norm_std):
    """
    Convert samples from normalized to physical space and compute summaries.

    Parameters
    ----------
    samples_norm : (N, L) generated curves in normalized space
    norm_mean    : (L,) per-grid-point mean
    norm_std     : (L,) per-grid-point std

    Returns
    -------
    samples_phys : (N, L) physical c_s^2 curves
    mean_curve   : (L,) point-wise mean
    std_curve    : (L,) point-wise std
    median_curve : (L,) point-wise median
    q16, q84     : (L,) 16th / 84th percentiles
    """
    samples_phys = samples_norm * norm_std.unsqueeze(0) + norm_mean.unsqueeze(0)

    mean_curve   = samples_phys.mean(dim=0)
    std_curve    = samples_phys.std(dim=0)
    median_curve = samples_phys.median(dim=0).values
    q16          = samples_phys.quantile(0.16, dim=0)
    q84          = samples_phys.quantile(0.84, dim=0)

    return samples_phys, mean_curve, std_curve, median_curve, q16, q84


def filter_causal(samples_phys, cs2_min=0.0, cs2_max=1.0):
    """
    Keep only curves satisfying cs2_min <= c_s^2 <= cs2_max at every grid point.
    Returns the filtered (N', L) tensor.
    """
    keep = ((samples_phys >= cs2_min) & (samples_phys <= cs2_max)).all(dim=1)
    n_before = samples_phys.shape[0]
    filtered = samples_phys[keep]
    print(f"  Causality filter {cs2_min} <= c_s^2 <= {cs2_max}: "
          f"kept {filtered.shape[0]}/{n_before} "
          f"({100 * filtered.shape[0] / max(n_before, 1):.1f}%)")
    return filtered


def summarize(samples_phys):
    """Recompute summary statistics on an arbitrary batch of curves."""
    if samples_phys.shape[0] == 0:
        L = samples_phys.shape[1] if samples_phys.ndim == 2 else 0
        nan_L = torch.full((L,), float('nan'))
        return nan_L, nan_L, nan_L, nan_L, nan_L

    if samples_phys.shape[0] == 1:
        row = samples_phys[0]
        zero_L = torch.zeros_like(row)
        return row, zero_L, row, row, row

    return (
        samples_phys.mean(dim=0),
        samples_phys.std(dim=0),
        samples_phys.median(dim=0).values,
        samples_phys.quantile(0.16, dim=0),
        samples_phys.quantile(0.84, dim=0),
    )