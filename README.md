# Generative artificial intelligence for reconstructing neutron-star matter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22982277.svg)](https://doi.org/10.5281/zenodo.22982277)

Code for the paper *Generative artificial intelligence for reconstructing
neutron-star matter* (J. Yu. Panteleeva, H. Alharazin, E. Epelbaum,
Ruhr-Universität Bochum).

We reconstruct the squared speed of sound $c_s^2(n_B)$ of cold, charge-neutral,
$\beta$-equilibrated neutron-star matter using a denoising-diffusion model as a
prior over the shape of the equation of state. Low-density chiral-effective-field-theory
input is built in while the model draws a curve; the perturbative-QCD and the
astrophysical measurements are applied afterwards by reweighting the drawn
samples. Because the network is the only trained part, the prior and the data
stay separate, and new measurements update the result by reweighting alone, with
no retraining.

---

## Data availability

The complete training and validation data (classes 1–13), all trained models with training logs, and the posterior samples from the paper are archived on Zenodo: [doi:10.5281/zenodo.22982277](https://doi.org/10.5281/zenodo.22982277) (CC-BY 4.0). Everything in the pipeline below can be reproduced from scratch, or downloaded there instead.

---

## System requirements

- **OS**: tested on Linux (HPC; training and sampling) and macOS (Apple Silicon; analysis). Any OS with a standard Python ≥ 3.12 installation should work.
- **Python**: 3.12 (tested with 3.12.12).
- **Dependencies**: pinned in [`requirements.txt`](requirements.txt) — `numpy` 2.4.6, `scipy` 1.17.1, `matplotlib` 3.10.8, `torch` 2.6.0 (CUDA 12.4 build on the cluster), `joblib` 1.5.3, `pandas` 3.0.1, `jupyter`/`ipython` (analysis notebook only).
- **Hardware**: no non-standard hardware is required for the demo or the analysis — a normal desktop/laptop is sufficient. *Training* the diffusion prior and the full production runs were performed on one NVIDIA A100 GPU (SLURM job with 1 GPU, 8 CPU cores): ≈ 12 h per trained model, and ≈ 15 h per 100 000-sample production run (sampling + stellar-structure solving + reweighting). The code falls back to CPU automatically (`torch.cuda.is_available()`), just slower. Since all trained models and posteriors are provided (see Data availability), a GPU is only needed to redo these stages from scratch.

---

## Installation

```bash
git clone https://github.com/JuliaYu24/Neutron_Star_Matter_Diffusion.git
cd Neutron_Star_Matter_Diffusion
pip install -r requirements.txt
```

Typical install time on a normal desktop computer: **a few minutes** (dominated by the PyTorch download). No compilation and no further setup are required; external data inputs are described in [`README_EXTERNAL_DATA.md`](README_EXTERNAL_DATA.md).

---

## Demo — reproduce the paper's results (no GPU, no retraining)

The published posterior samples on Zenodo make the full analysis reproducible on a normal desktop computer without rerunning the expensive stages:

1. Download `posteriors.zip` from [doi:10.5281/zenodo.22982277](https://doi.org/10.5281/zenodo.22982277) and unpack it.
2. Open `analysis/notebook_diagnostics_kde.ipynb` and set `OUTPUT_PATH` to the path of the posterior you are interested in.
3. Run the notebook top to bottom.

**Expected output**: all numbers and figures of the paper, displayed in the notebook.

**Expected runtime**: ≈ 20 minutes per posterior on a normal desktop computer (tested on an Apple-Silicon MacBook; no GPU required).

To instead re-generate a posterior from the shipped trained model, run `run_sampling.py` (reduce `n_samples` in its config block from 100 000 for a quick check); the full pipeline from scratch is described under "Running the full pipeline" below.

---

## Guides for each part

This file is the overview. Each part of the project has its own guide:

| Guide | What it covers |
|-------|----------------|
| [`README_EXTERNAL_DATA.md`](README_EXTERNAL_DATA.md) | **Start here.** The inputs you need to download, where each file goes, and their sources. |
| [`README_TRAINING.md`](README_TRAINING.md) | Training the diffusion prior with `run_training.py` / `eos_diffusion/train.py`, and reproducing the published model $M_0$. |
| [`README_SAMPLING.md`](README_SAMPLING.md) | Drawing samples, solving the stellar-structure equations, and reweighting (`run_sampling.py` → `apply_kde_nicer.py` → optional `apply_heavy_mass_constraint.py`), with the full list of runs. |
| [`analysis/README_ANALYSIS.md`](analysis/README_ANALYSIS.md) | Turning a posterior into the paper's numbers, diagnostics, tables and figures via `notebook_diagnostics_kde.ipynb`. |

---

## Repository layout

```
.
├── eos_training_curves/         # build the 10-class training set of c_s^2 curves
├── eos_class_validation_curves/ # extra families (classes 11–13) used in the robustness runs
├── eos_diffusion/               # the diffusion model (architecture, noise schedule, training, sampling)
├── eos_sampling/                # sampling, the stellar-structure + tidal solver, likelihoods, reweighting, pQCD
├── analysis/                    # derived quantities, diagnostics, figures, and the driver notebook
│   └── chEFT/                   # chiral-EFT anchor extraction (rebuilds the anchor .npy files)
│
├── run_training.py              # train the prior
├── run_sampling.py              # sample + reweight  →  posterior .pt
├── apply_kde_nicer.py           # switch the NICER term to the higher-fidelity KDE likelihood
├── apply_heavy_mass_constraint.py # optional: add a heavy-mass constraint by reweighting
├── jackknife_errors.py          # jackknife resampling for error estimates
│
├── README_EXTERNAL_DATA.md      # inputs and where they go  (see table above)
├── README_TRAINING.md
├── README_SAMPLING.md
└── analysis/README_ANALYSIS.md
```

---

## Running the full pipeline

The steps run in order; each one links to its guide.

**0. Get the inputs** — see [`README_EXTERNAL_DATA.md`](README_EXTERNAL_DATA.md).

**1. Rebuild the chiral-EFT anchors** — [`README_EXTERNAL_DATA.md`](README_EXTERNAL_DATA.md) §1–2
```bash
cd analysis/chEFT
python cs2_betaeq_anchors.py --outdir .     # writes the anchor / reference-point .npy files
```

**2. Train the diffusion prior** — [`README_TRAINING.md`](README_TRAINING.md)
```bash
python run_training.py                       # writes checkpoints/eos_ddpm_best.pt
```

> You do not need to run this. The four trained models used in the paper are
> already in `checkpoints/`: `eos_ddpm_best_base_line.pt` (baseline, $M_0$) and
> `eos_ddpm_best_11.pt`, `eos_ddpm_best_12.pt`, `eos_ddpm_best_13.pt`
> ($M_\mathrm{I}$–$M_\mathrm{III}$, baseline plus class 11, 12 and 13), each the
> best-epoch snapshot. Train only if you want to rebuild them.

**3. Sample and reweight** — [`README_SAMPLING.md`](README_SAMPLING.md)
```bash
python run_sampling.py                        # sampling + stellar structure + reweighting  →  posterior .pt
python apply_kde_nicer.py                      # switch the NICER term to the KDE likelihood
```

**4. Diagnostics, tables and figures** — [`analysis/README_ANALYSIS.md`](analysis/README_ANALYSIS.md)
Open `analysis/notebook_diagnostics_kde.ipynb`, set `OUTPUT_PATH` to your
posterior `.pt`, and run it from top to bottom.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Panteleeva:2026zxs,
    author = "Panteleeva, Julia Yu. and Alharazin, Herzallah and Epelbaum, Evgeny",
    title = "{Generative artificial intelligence for reconstructing neutron-star matter}",
    eprint = "2608.17457",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "8",
    year = "2026"
}
```

---

## Contact

Correspondence: J. Yu. Panteleeva — panteleevajuly@gmail.com
