"""
Training Loop
-------------
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .schedule import CosineSchedule
from .model import EOSDiffusionNet
from .diffusion import VPredictionDDPM
from .ema import EMA
def get_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(
    data_path:       str   = "../eos_training_curves/data",
    epochs:          int   = 100,
    batch_size:      int   = 256,
    n_train_per_class: int = None,
    n_val_per_class:   int = None,
    lr:              float = 1e-4,
    weight_decay:    float = 1e-4,
    grad_clip:       float = 1.0,
    ema_decay:       float = 0.9999,
    T:               int   = 1000,
    warmup_epochs:   int   = 2,
    val_fraction:    float = 0.01,
    save_every:      int   = 25,
    use_amp:         bool  = True,
    checkpoint_dir:  str   = "checkpoints",
    resume_from:     str   = None,
    seed:            int   = 42,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        print(f"  cuDNN benchmark: enabled")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_amp and device.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print(f"  AMP: bfloat16 (native)")
        else:
            amp_dtype = torch.float16
            print(f"  AMP: float16 (with GradScaler)")
    else:
        use_amp = False
        amp_dtype = torch.float32
        print(f"  AMP: disabled")

    use_scaler = use_amp and (amp_dtype == torch.float16)
    import glob
    import numpy as np
    '''
    def _load_split(split):
        files = sorted(glob.glob(os.path.join(data_path, split, "*", "*_data.npy")))
        if not files:
            raise FileNotFoundError(f"No *_data.npy under {os.path.join(data_path, split)}")
        arrs = [np.load(f) for f in files]                      # 10 class files
        return torch.from_numpy(np.concatenate(arrs, axis=0)).float()
    '''
    def _load_split(split, n_per_class=None):
        files = sorted(glob.glob(os.path.join(data_path, split, "*", "*_data.npy")))
        if not files:
            raise FileNotFoundError(f"No *_data.npy under {os.path.join(data_path, split)}")
        arrs = []
        for f in files:
            a = np.load(f)
            if n_per_class is not None:
                if a.shape[0] < n_per_class:
                    print(f"  WARNING {f}: only {a.shape[0]} < {n_per_class}")
                a = a[:n_per_class]
            print(f"  {os.path.basename(f)}: using {a.shape[0]}")
            arrs.append(a)
        return torch.from_numpy(np.concatenate(arrs, axis=0)).float()
    print(f"\nLoading raw curves from {data_path}/(train|val) ...")
    X_train_raw = _load_split("train")
    X_val_raw   = _load_split("val")
    norm_mean = X_train_raw.mean(dim=0)
    norm_std  = X_train_raw.std(dim=0).clamp_min(1e-8)

    X_train = (X_train_raw - norm_mean) / norm_std
    X_val   = (X_val_raw   - norm_mean) / norm_std
    del X_train_raw, X_val_raw

    norm_mean = norm_mean.cpu()
    norm_std  = norm_std.cpu()
    nB_grid   = torch.linspace(0.5, 8.0, X_train.shape[1])

    n_train = X_train.shape[0]
    n_val   = X_val.shape[0]

    print(f"  Train: {n_train}   Val: {n_val}")
    print(f"  Normalized train range: [{X_train.min():.3f}, {X_train.max():.3f}]  "
          f"mean {X_train.mean():.4f}  std {X_train.std():.4f}")
    is_cuda    = (device.type == 'cuda')
    nw_train   = 4 if is_cuda else 0
    nw_val     = 2 if is_cuda else 0
    pin        = is_cuda

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size,
                              shuffle=True, num_workers=nw_train, pin_memory=pin,
                              drop_last=True,
                              persistent_workers=(nw_train > 0))
    val_loader   = DataLoader(TensorDataset(X_val), batch_size=batch_size,
                              shuffle=False, num_workers=nw_val, pin_memory=pin,
                              drop_last=False,
                              persistent_workers=(nw_val > 0))

    grid_size = X_train.shape[1]
    schedule  = CosineSchedule(T=T)
    config = {
        'grid_size':     grid_size,
        'hidden_dim':    256,
        'kernel_size':   7,
        'n_res_blocks':  12,
        'n_groups':      3,
        'n_heads':       4,
        'dropout':       0.1,
        'T':             T,
        'epochs':        epochs,
        'batch_size':    batch_size,
        'lr':            lr,
        'weight_decay':  weight_decay,
        'ema_decay':     ema_decay,
        'warmup_epochs': warmup_epochs,
        'amp_dtype':     str(amp_dtype),
        'seed':          seed,
    }

    model = EOSDiffusionNet(
        grid_size    = config['grid_size'],
        hidden_dim   = config['hidden_dim'],
        kernel_size  = config['kernel_size'],
        n_res_blocks = config['n_res_blocks'],
        n_groups     = config['n_groups'],
        n_heads      = config['n_heads'],
        dropout      = config['dropout'],
    ).to(device)
    diffusion = VPredictionDDPM(model, schedule, device)
    ema       = EMA(model, decay=ema_decay)

    n_params = count_parameters(model)
    print(f"\n  Model parameters: {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Grid size: {grid_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    print(f"  Steps/epoch: {steps_per_epoch}   Total steps: {total_steps}   Warmup steps: {warmup_steps}")

    lr_lambda = get_lr_lambda(warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    start_epoch = 1
    global_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n  Resuming from {resume_from}...")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        if use_scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', (start_epoch - 1) * steps_per_epoch)
        print(f"  Resumed at epoch {start_epoch}, global step {global_step}")

    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr\n")

    print(f"\n{'='*70}")
    print(f"  Starting training for {epochs} epochs (from epoch {start_epoch})")
    print(f"  Batch size: {batch_size}   Diffusion steps T: {T}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for (x0_batch,) in train_loader:
            x0_batch = x0_batch.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.training_step(x0_batch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            scheduler.step()

            total_loss += loss.item()
            n_batches  += 1
            global_step += 1

        avg_train_loss = total_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        val_loss = 0.0
        n_val_batches = 0
        ema.apply(model)
        model.eval()
        with torch.no_grad():
            for (x0_val,) in val_loader:
                x0_val = x0_val.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    vl = diffusion.training_step(x0_val)
                val_loss += vl.item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)
        ema.restore(model)

        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            marker = "  ★ best"

        print(f"Epoch {epoch:3d}/{epochs}  |  "
              f"Train: {avg_train_loss:.6f}  |  "
              f"Val: {avg_val_loss:.6f}  |  "
              f"LR: {current_lr:.2e}{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.8f},{avg_val_loss:.8f},{current_lr:.2e}\n")

        if epoch % save_every == 0 or epoch == epochs:
            checkpoint = {
                'epoch':       epoch,
                'global_step': global_step,
                'model_state': model.state_dict(),
                'ema':         ema.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'scheduler':   scheduler.state_dict(),
                'scaler':      scaler.state_dict() if use_scaler else None,
                'train_loss':  avg_train_loss,
                'val_loss':    avg_val_loss,
                'config':      config,
            }
            path = os.path.join(checkpoint_dir, f"eos_ddpm_epoch{epoch:03d}.pt")
            torch.save(checkpoint, path)
            print(f"  → Saved checkpoint: {path}")

        if marker:
            best_path = os.path.join(checkpoint_dir, "eos_ddpm_best.pt")
            ema.apply(model)
            best_save = {
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'config':      config,
                'val_loss':    avg_val_loss,
                'schedule': {
                    'T': schedule.T,
                    'alpha_bar':                schedule.alpha_bar.cpu(),
                    'sqrt_alpha_bar':           schedule.sqrt_alpha_bar.cpu(),
                    'sqrt_one_minus_alpha_bar': schedule.sqrt_one_minus_alpha_bar.cpu(),
                    'alpha':                    schedule.alpha.cpu(),
                    'beta':                     schedule.beta.cpu(),
                    'posterior_variance':       schedule.posterior_variance.cpu(),
                },
            }
            if norm_mean is not None and norm_std is not None:
                best_save['normalization'] = {
                    'mean': norm_mean,
                    'std':  norm_std,
                }
            if nB_grid is not None:
                best_save['nB_over_n0_grid'] = nB_grid
            torch.save(best_save, best_path)
            ema.restore(model)
            print(f"  → Saved best model: {best_path}")

    ema.apply(model)
    final_save = {
        'epoch':       epochs,
        'model_state': model.state_dict(),
        'config':      config,
        'schedule': {
            'T': schedule.T,
            'alpha_bar':                schedule.alpha_bar.cpu(),
            'sqrt_alpha_bar':           schedule.sqrt_alpha_bar.cpu(),
            'sqrt_one_minus_alpha_bar': schedule.sqrt_one_minus_alpha_bar.cpu(),
            'alpha':                    schedule.alpha.cpu(),
            'beta':                     schedule.beta.cpu(),
            'posterior_variance':       schedule.posterior_variance.cpu(),
        },
    }
    if norm_mean is not None and norm_std is not None:
        final_save['normalization'] = {
            'mean': norm_mean,
            'std':  norm_std,
        }
    if nB_grid is not None:
        final_save['nB_over_n0_grid'] = nB_grid
    final_path = os.path.join(checkpoint_dir, "eos_ddpm_final.pt")
    torch.save(final_save, final_path)
    print(f"\nTraining complete. Final EMA model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    ema.restore(model)