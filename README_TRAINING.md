# Training the diffusion model

The denoising-diffusion prior over the squared sound speed \(c_s^2(n_B)\) is
trained by `eos_diffusion/train.py`. The entry point is `run_training.py`, which
calls `train(...)` with the paths used for the published model; all training
options are exposed as keyword arguments of `train()` (there is no command-line
parser — set options by editing the `train(...)` call in `run_training.py`, or by
importing `train` and passing keywords).

## Input data

Training reads standardized curves from two sibling folders,
```
<data_path>/train/<class>/*_data.npy
<data_path>/val/<class>/*_data.npy
```
globbing every class sub-folder (see `eos_training_curves/` for how these curves
are generated, with seeds). Per-grid-point mean and standard deviation are
computed from the **training** split and applied to both splits; these statistics
are stored inside the saved model so that sampling recovers physical units.

## Quick start

From the repository root:
```
python run_training.py
```
This trains for 100 epochs from `eos_training_curves/data`, writing to
`checkpoints/`. To change any option, edit the `train(...)` call in
`run_training.py`, e.g.
```python
from eos_diffusion.train import train
train(
    data_path      = "eos_training_curves/data",
    epochs         = 100,
    batch_size     = 120,
    save_every     = 25,
    checkpoint_dir = "checkpoints",
    resume_from    = None,
)
```

## Options (`train()` keyword arguments)

| Argument | Default | Meaning |
|---|---|---|
| `data_path` | `"../eos_training_curves/data"` | Root holding `train/` and `val/` (the launcher overrides this to `eos_training_curves/data`). |
| `epochs` | `100` | Total epochs; also sets the cosine learning-rate horizon. |
| `batch_size` | `256` | Mini-batch size. |
| `lr` | `1e-4` | Peak AdamW learning rate. |
| `weight_decay` | `1e-4` | AdamW weight decay. |
| `grad_clip` | `1.0` | Global gradient-norm clip. |
| `warmup_epochs` | `2` | Linear LR warm-up before per-step cosine decay to zero. |
| `ema_decay` | `0.9999` | Exponential-moving-average decay; validation and all saved weights use the EMA. |
| `T` | `1000` | Number of diffusion steps (cosine noise schedule). |
| `save_every` | `25` | Write a full (resumable) checkpoint every N epochs; a checkpoint is also written on the final epoch. |
| `checkpoint_dir` | `"checkpoints"` | Output directory for checkpoints and the training log. |
| `resume_from` | `None` | Path to a full epoch checkpoint to continue from (see below). |
| `use_amp` | `True` | Mixed precision: bfloat16 on capable GPUs, else float16 with gradient scaling; falls back to float32 on CPU. |
| `seed` | `42` | Seed for `torch`/CUDA RNGs (reproducibility). |

The network architecture is fixed in the `config` block of `train.py`
(hidden width 256, twelve residual blocks in three attention-separated groups,
four attention heads, kernel size 7, dropout 0.1; \(\approx\!1.94\times10^{7}\)
parameters) and is recorded inside every checkpoint. Change it there, not through
`train()`.

> Not active as shipped: `val_fraction` is unused (validation is taken from the
> `val/` folder, not split from training), and `n_train_per_class` /
> `n_val_per_class` are accepted but not applied by the loader. Ignore these
> unless you wire them in.

## Outputs (written to `checkpoint_dir`)

| File | When | Contents | Use |
|---|---|---|---|
| `training_log.csv` | every epoch | `epoch, train_loss, val_loss, lr` | monitoring (overwritten only on a fresh run) |
| `eos_ddpm_epoch<NNN>.pt` | every `save_every` epochs + final | model, EMA, optimizer, scheduler, scaler, config, losses, epoch, global step | **resuming** |
| `eos_ddpm_best.pt` | whenever validation loss improves | EMA weights, schedule, normalization, density grid | **inference** |
| `eos_ddpm_final.pt` | end of run | EMA weights, schedule, normalization, density grid | **inference** |

Only the `eos_ddpm_epoch<NNN>.pt` files carry optimizer/scheduler state and can
be resumed; `best`/`final` are inference-ready EMA snapshots consumed by
`eos_diffusion/inference.py` and the sampling pipeline.

## Continuing a run

Point `resume_from` at a full epoch checkpoint:
```python
train(epochs=100, resume_from="checkpoints/eos_ddpm_epoch075.pt")
```
This restores the model, optimizer, scheduler, EMA (and the float16 gradient
scaler when used) and continues from the saved epoch + 1, appending to the
existing `training_log.csv`. Because the cosine learning-rate horizon is
`epochs × steps_per_epoch`, **pass the same `epochs`, `batch_size`, and
learning-rate settings as the original run** so the schedule lines up; resuming
with a different `epochs` redefines the decay curve. Resume only from an
`eos_ddpm_epoch<NNN>.pt` file — `best`/`final` do not contain optimizer state.

## Reproducing the published model

The baseline model \(M_0\) was trained for 100 epochs with `batch_size=120`,
`lr=1e-4`, `weight_decay=1e-4`, `ema_decay=0.9999`, `warmup_epochs=2`, `T=1000`,
and a fixed seed; the fiducial weights are the lowest-validation-loss EMA snapshot
(`eos_ddpm_best.pt`). Set `batch_size=120` to match the published run (the shipped
default is 256).
