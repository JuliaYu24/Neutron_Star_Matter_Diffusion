# Generative artificial intelligence for reconstructing neutron-star matter

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

```

```

---

## Contact

Correspondence: J. Yu. Panteleeva — panteleevajuly@gmail.com
```
