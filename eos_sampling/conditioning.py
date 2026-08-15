"""
Build inpainting conditioning tensors from physical (nB, cs2, sigma) points.

The anchor error may be supplied either as K independent sigmas
(cs2_sigma) or as a full K x K covariance matrix (cs2_cov).  When cs2_cov
is given, the sampler draws correlated Gaussian perturbations at the
anchor points, which better represents chi-EFT truncation uncertainty
(highly correlated across density) than independent draws.
"""

import torch


def build_conditioning(nB_grid, nB_known, cs2_known,
                       norm_mean, norm_std,
                       cs2_sigma=None, cs2_cov=None):
    """
    Map physical known-point data to normalized tensors on the training grid.

    Parameters
    ----------
    nB_grid    : (L,) full dimensionless nB/n0 grid from the checkpoint
    nB_known   : (K,) known nB/n0 values (chi-EFT anchor points)
    cs2_known  : (K,) known c_s^2 central values in physical units
    norm_mean  : (L,) per-grid-point training mean
    norm_std   : (L,) per-grid-point training std
    cs2_sigma  : (K,) independent experimental errors, or None
    cs2_cov    : (K, K) FULL covariance matrix in physical units, or None.
                 If given, takes precedence over cs2_sigma.

    Returns
    -------
    mask         : (L,) binary, 1 at conditioned grid points
    x_cond       : (L,) normalized central values, 0 elsewhere
    sigma_cond   : (L,) normalized independent errors, 0 elsewhere, or None.
                   Populated only if cs2_sigma was given AND cs2_cov was not.
    indices      : list[int] of matched grid indices
    anchor_chol  : (K, K) Cholesky factor of the covariance in NORMALIZED
                   space, or None if cs2_cov was not given.

    Usage in run_sampling.py
    ------------------------
    Independent errors:
        cs2_sigma = torch.tensor([0.008, 0.035, 0.07])
        cs2_cov   = None

    Correlated errors:
        sigma     = torch.tensor([0.008, 0.035, 0.07])
        # Example: 60% correlation between adjacent anchors,
        # 30% between far-apart anchors.
        corr      = torch.tensor([[1.0, 0.6, 0.3],
                                  [0.6, 1.0, 0.6],
                                  [0.3, 0.6, 1.0]])
        cs2_cov   = sigma.unsqueeze(0) * sigma.unsqueeze(1) * corr
        cs2_sigma = None   # or ignored
    """
    L = nB_grid.shape[0]
    mask       = torch.zeros(L)
    x_cond     = torch.zeros(L)
    sigma_cond = torch.zeros(L)
    indices    = []

    grid_spacing = (nB_grid[1] - nB_grid[0]).abs().item()

    for k in range(len(nB_known)):
        diffs = (nB_grid - nB_known[k]).abs()
        j = int(diffs.argmin().item())

        if diffs[j].item() > 0.5 * grid_spacing:
            print(f"  WARNING: nB_known={nB_known[k]:.4f} matched to "
                  f"nB_grid[{j}]={nB_grid[j]:.4f} (delta={diffs[j]:.4f})")

        x_cond[j] = (cs2_known[k] - norm_mean[j]) / norm_std[j]
        mask[j]   = 1.0
        indices.append(j)

        # sigma_cond is filled only when cs2_sigma is given AND cs2_cov is not
        if cs2_sigma is not None and cs2_cov is None:
            sigma_cond[j] = cs2_sigma[k] / norm_std[j]

    n_known = int(mask.sum().item())
    print(f"  Conditioning: {n_known} known points out of {L} grid points "
          f"({100 * n_known / L:.1f}%)")

    # ------- Covariance path -------
    anchor_chol = None
    if cs2_cov is not None:
        K   = len(indices)
        cov = torch.as_tensor(cs2_cov, dtype=torch.float32)
        if cov.shape != (K, K):
            raise ValueError(
                f"cs2_cov must be (K, K) with K={K}, got {tuple(cov.shape)}")

        # Scale each row/col by 1 / norm_std at the corresponding anchor index
        # so the Cholesky is of the covariance in NORMALIZED space.
        std_at_anchors = torch.tensor(
            [float(norm_std[indices[i]]) for i in range(K)],
            dtype=torch.float32)
        D_inv = 1.0 / std_at_anchors
        cov_norm = cov * D_inv.unsqueeze(1) * D_inv.unsqueeze(0)
        # Ridge for numerical PSD safety
        cov_norm = cov_norm + 1e-8 * torch.eye(K)
        anchor_chol = torch.linalg.cholesky(cov_norm)
        print(f"  Covariance-based anchor jitter enabled (K={K}); "
              f"sigma_cond is set to zero and anchor_chol will be used.")
        sigma_cond = torch.zeros(L)

    # Return None for sigma_cond if neither cs2_sigma nor cs2_cov was supplied
    if cs2_sigma is None and cs2_cov is None:
        sigma_out = None
    else:
        sigma_out = sigma_cond

    return mask, x_cond, sigma_out, indices, anchor_chol