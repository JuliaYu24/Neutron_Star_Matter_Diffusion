"""
Load Model for Interference

Load a trained EOS diffusion model from checkpoint.

Works with:
   - "eos_ddpm_final.pt"    (final EMA weights + schedule)
   - "eos_ddpm_best.pt"     (best EMA weights + schedule)
   - "eos_ddm_epochNNN.pt"  (epoch checkpoint, extracts EMA weights)

Usage:
    from eos_diffusion.inference import load_model
    from eos_diffusion.diffusion import VPredictionDDPM

    model, schedule, config, norm, grid = load_model("checkpoints/eos_ddpm_best.pt")

    # Single forward pass
    with torch.no_grad():
        v_pred = model(x_noisy, t, mask, x_cond)

    # Or wrap VPredictionDDPM for full sampling
    diffusion = VPredictionDDPM(model, schedule, device)
    x0_hat = diffusion.predict_x0_from_v(x_t, v_pred, t)

    # De-normalize to physical c_s values
    if norm is not None:
        cs2_physical = x0_hat * norm['std'].to(device) + norm['mean'].to(device)

    # 'grid' is the dimensionless nb/n0 axis the model was trained on.
"""

import torch

from .schedule import CosineSchedule
from .model import EOSDiffusionNet

def load_model(checkpoint_path: str, device: torch.device = None):
    """
    Args:
        checkpoint_path : path to .pt checkpoint file
        device          : torch device (default: cuda if available, else cpu)

    Returns:
        model    : EOSDiffusionNet with EMA weights loaded, in eval mode
        schedule : CosineSchedule on the correct device
        config   : dict of hyperparameters used during training
        norm     : dict with "mean" and "std" tensors, or None if not stored
        grid     : tensor (N,) of dimensionless nB/n0 grid points, or None
                   if not stored (older checkpoints)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt['config']
    print(f" Config: grid_size= {config['grid_size']}, "
          f"hidden_dim = {config['hidden_dim']}, "
          f"n_res_blocks={config['n_res_blocks']}, "
          f"T = {config['T']}")
    model = EOSDiffusionNet(
        grid_size    = config['grid_size'],
        hidden_dim   = config['hidden_dim'],
        kernel_size  = config['kernel_size'],
        n_res_blocks = config['n_res_blocks'],
        n_groups     = config['n_groups'],
        n_heads      = config['n_heads'],
        dropout      = 0.0,
    ).to(device)

    if 'ema' in ckpt and 'schedule' not in ckpt:
        print("  Loading EMA weights from epoch checkpoint...")
        ema_shadow = ckpt['ema']['shadow']
        state_dict = ckpt['model_state'].copy()
        for name in ema_shadow:
            state_dict[name] = ema_shadow[name]
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt['model_state'])

    model.eval()

    T = config['T']
    if 'schedule' in ckpt:
        schedule = CosineSchedule(T=T)
        for key in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar',
                     'alpha', 'beta', 'posterior_variance']:
            setattr(schedule, key, ckpt['schedule'][key])
        schedule.to(device)
    else:
        schedule = CosineSchedule(T=T)
        schedule.to(device)

    norm = None
    if 'normalization' in ckpt:
        norm = ckpt['normalization']
        print(f"  Normalization: mean {norm['mean'].shape}, std {norm['std'].shape}")
    else:
        print(f"  WARNING: No normalization statistics in checkpoint.")
        print(f"           Load mean/std separately for de-normalization.")

    grid = ckpt.get('nB_over_n0_grid', None)
    if grid is not None:
        print(f"  Physical grid: nB_over_n0_grid {tuple(grid.shape)}"
              f"  range [{grid.min():.3f}, {grid.max():.3f}]")
    else:
        print(f"  NOTE: No nB_over_n0_grid in checkpoint "
              f"(older checkpoint; derive from endpoints if needed).")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters ({n_params/1e6:.2f}M)")
    print(f"  Device: {device}")

    if 'val_loss' in ckpt:
        print(f"  Val loss at save: {ckpt['val_loss']:.6f}")
    if 'epoch' in ckpt:
        print(f"  Trained for {ckpt['epoch']} epochs")

    print(f"  Ready for inference.\n")
    return model, schedule, config, norm, grid