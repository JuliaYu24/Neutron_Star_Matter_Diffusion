"""
Base DDPM reverse sampler with inpainting + replacement.

Generates samples from the trained diffusion prior, with chi-EFT
anchor points enforced as inpainting conditioning.
Astrophysical conditioning is handled separately by the
importance reweighter (see eos_sampling.reweighting).

Supports optional correlated anchor-error jitter via a Cholesky
factor (anchor_chol) plus anchor grid indices.  When not supplied,
the sampler falls back to per-point independent jitter using
sigma_cond.
"""

import torch


@torch.no_grad()
def sample_ddpm(model, schedule, mask, x_cond,
                sigma_cond=None,
                anchor_chol=None, anchor_indices=None,
                n_samples=60, batch_size=2,
                device=None, seed=None, x0_clamp=5.0):
    """
    Generate `n_samples` EOS curves from the trained DDPM, conditioned
    on chi-EFT anchor points via inpainting replacement.

    Parameters
    ----------
    model          : trained EOSDiffusionNet (v-prediction denoiser)
    schedule       : CosineSchedule with all diffusion tensors precomputed
    mask           : (L,) binary, 1 at conditioned grid points
    x_cond         : (L,) normalized central values at conditioned points
    sigma_cond     : (L,) normalized independent errors, or None.
                     Ignored if anchor_chol is given.
    anchor_chol    : (K, K) Cholesky factor of anchor covariance in
                     normalized space, or None.
    anchor_indices : (K,) long-tensor of grid indices where the K
                     anchors live.  Required when anchor_chol is given.
    n_samples      : total curves to generate
    batch_size     : curves processed simultaneously per GPU pass
    device         : torch device; defaults to the model's device
    seed           : RNG seed for reproducibility
    x0_clamp       : hard clamp on the Tweedie x0_hat estimate

    Returns
    -------
    samples_norm : (n_samples, L) generated curves in normalized space
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    L = mask.shape[0]
    T = schedule.T

    mask_batch  = mask.unsqueeze(0).to(device)
    x_cond_base = x_cond.unsqueeze(0).to(device)

    # Choose jitter mode once
    use_cov_jitter = anchor_chol is not None
    if use_cov_jitter:
        if anchor_indices is None:
            raise ValueError("anchor_chol requires anchor_indices.")
        anchor_chol    = anchor_chol.to(device)
        anchor_indices = anchor_indices.to(device).long()
        K = anchor_chol.shape[0]
        sigma_cond_batch = None
    else:
        sigma_cond_batch = (sigma_cond.unsqueeze(0).to(device)
                            if sigma_cond is not None else None)
        K = None

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    all_x0 = []
    n_generated = 0

    while n_generated < n_samples:
        B = min(batch_size, n_samples - n_generated)

        x_t              = torch.randn(B, L, device=device)
        mask_b           = mask_batch.expand(B, -1)
        one_minus_mask_b = 1.0 - mask_b

        # --- anchor jitter ---
        if use_cov_jitter:
            eps_k  = torch.randn(B, K, device=device)
            delta_k = eps_k @ anchor_chol.t()           # (B, K) ~ N(0, cov_norm)
            jitter = torch.zeros(B, L, device=device)
            jitter[:, anchor_indices] = delta_k
            x_cond_b = x_cond_base.expand(B, -1) + jitter
        elif sigma_cond_batch is not None:
            jitter   = torch.randn(B, L, device=device) * sigma_cond_batch
            x_cond_b = x_cond_base.expand(B, -1) + jitter
        else:
            x_cond_b = x_cond_base.expand(B, -1)

        with torch.amp.autocast(device_type, enabled=False):
            for tau in range(T, 0, -1):
                t_tensor = torch.full((B,), tau, device=device, dtype=torch.float32)

                v_pred = model(x_t, t_tensor, mask_b, x_cond_b)

                sqrt_abar    = schedule.sqrt_alpha_bar[tau]
                sqrt_1m_abar = schedule.sqrt_one_minus_alpha_bar[tau]
                x0_hat = sqrt_abar * x_t - sqrt_1m_abar * v_pred
                x0_hat = x0_hat.clamp(-x0_clamp, x0_clamp)

                alpha_bar_prev = schedule.alpha_bar[tau - 1]
                alpha_bar_curr = schedule.alpha_bar[tau]
                beta_t         = schedule.beta[tau - 1]
                alpha_t        = schedule.alpha[tau - 1]

                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_curr)
                coeff_xt = (alpha_t.sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_curr)
                mu = coeff_x0 * x0_hat + coeff_xt * x_t

                if tau > 1:
                    sigma = schedule.posterior_variance[tau - 1].sqrt()
                    z     = torch.randn_like(x_t)
                    x_t   = mu + sigma * z

                    sqrt_abar_prev = schedule.sqrt_alpha_bar[tau - 1]
                    sqrt_1m_prev   = schedule.sqrt_one_minus_alpha_bar[tau - 1]
                    noise_rep      = torch.randn_like(x_t)
                    x_known_noised = sqrt_abar_prev * x_cond_b + sqrt_1m_prev * noise_rep
                    x_t = mask_b * x_known_noised + one_minus_mask_b * x_t
                else:
                    x_t = mu
                    x_t = mask_b * x_cond_b + one_minus_mask_b * x_t

        all_x0.append(x_t.cpu())
        n_generated += B
        print(f"  Generated {n_generated}/{n_samples} samples",
              end="\r", flush=True)

    print()
    return torch.cat(all_x0, dim=0)