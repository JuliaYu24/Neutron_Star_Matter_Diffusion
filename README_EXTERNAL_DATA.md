# External data & code — download and placement guide

This repository does **not** include any third-party code, datasets, or large/
regenerable files. To reproduce the pipeline end to end you must obtain the items
below from their original sources and place them at the indicated paths. Nothing
here is redistributed — please cite the original works (see the end of this file).

All paths are **relative to the repository root** (the folder that contains
`eos_diffusion/`, `eos_sampling/`, `eos_training_curves/`,
`eos_class_validation_curves/`, and `analysis/`).

> **Note on the `.npy` files in `analysis/chEFT/`.** The arrays
> `cs2_BETAEQ_Lambda-*_anchor_*.npy`, `cs2_BETAEQ_Lambda-*_refpoint_*.npy`,
> `EA_SNM_*.npy`, `S2_BETAEQ_*.npy`, `L_BETAEQ_*.npy`, and `nB_density_*.npy`
> are **not** downloaded from anyone — they are produced by running
> `analysis/chEFT/cs2_betaeq_anchors.py` (Section 2). Regenerate them; do not
> look for them online.

---

## 0. Python packages (the BUQEYE chiral-EFT stack)

Needed to run `analysis/chEFT/cs2_betaeq_anchors.py` (the χEFT anchor extraction).

- **`nuclear_matter`** — from the BUQEYE *nuclear-matter-convergence* repository
  (MIT license). Provides the `nuclear_matter` package that
  `cs2_betaeq_anchors.py` imports.
  ```
  git clone https://github.com/buqeye/nuclear-matter-convergence
  cd nuclear-matter-convergence
  pip3 install .
  ```
  Alternatively, drop the repo's `nuclear_matter/` package folder into
  `analysis/chEFT/nuclear_matter/` (the script also finds it beside itself).
  Ref: Drischler, Meléndez, Furnstahl, Phillips, PRC **102**, 054315 (2020);
  Drischler, Furnstahl, Meléndez, Phillips, PRL **125**, 202702 (2020).
- **`gsum`** — `pip install gsum`  (source: https://github.com/buqeye/gsum)
- **`gptools`** — https://github.com/markchil/gptools  (or `pip install gptools`)

General runtime dependencies used across the code: `numpy`, `scipy`, `pandas`,
`matplotlib`, `torch`, `joblib`, `tables` (PyTables, for `pandas.read_hdf`),
`scikit-learn`, `seaborn`.

---

## 1. Chiral-EFT input tables (BUQEYE)  →  `analysis/chEFT/`

Both come from `github.com/buqeye/nuclear-matter-convergence`:

| File | Source path in repo | Place at | Needed for |
|------|--------------------|----------|-----------|
| `all_matter_data_high_density.csv` | `data/all_matter_data_high_density.csv` | `analysis/chEFT/all_matter_data_high_density.csv` | **Required** input to `cs2_betaeq_anchors.py` |
| `pressure_cs2_samples.csv` | `analysis/pressure_cs2_samples.csv` | `analysis/chEFT/pressure_cs2_samples.csv` | Only the PNM validation (`check_pnm_against_published`, Λ = 500) |

Direct (raw) links:
- https://raw.githubusercontent.com/buqeye/nuclear-matter-convergence/master/data/all_matter_data_high_density.csv
- https://raw.githubusercontent.com/buqeye/nuclear-matter-convergence/master/analysis/pressure_cs2_samples.csv

(The input CSV can also be pointed at via the `MATTER_CSV` environment variable
or the `--csv` flag instead of copying it into `analysis/chEFT/`.)

---

## 2. Regenerate the χEFT anchor / reference `.npy`  (not a download)

After Sections 0 and 1, produce the conditioning inputs the sampler reads:

```
cd analysis/chEFT
python cs2_betaeq_anchors.py --outdir .
```

With no `--Lambda` argument this processes **both** cutoffs (Λ = 500 and 450) and
writes, into `analysis/chEFT/`:

- `cs2_BETAEQ_Lambda-{500,450}_anchor_{nB,mean,cov,gridindex}_5pt.npy`
- `cs2_BETAEQ_Lambda-{500,450}_refpoint_{mean,sigma,nB}.npy`
- `EA_SNM_*`, `S2_BETAEQ_*`, `L_BETAEQ_*`, `nB_density_*` (`.npy`)

`run_sampling.py` reads the Λ = 500 `anchor_{nB,mean,cov}_5pt.npy` and
`refpoint_{mean,sigma,nB}.npy` from `analysis/chEFT/` (see its `ANCHOR_DIR`).

---

## 3. Perturbative-QCD marginalized likelihood (Komoltsev, Gorda, Kurkela)  →  `external/zenodo_15407795/`

- **Zenodo DOI 10.5281/zenodo.15407795** — "Marginalized QCD likelihood function"
  (use the latest version; currently V3, which fixed a results-affecting bug).
  https://zenodo.org/records/15407795
- Download **both** of:
  - `eos_marginalization.py`
  - `eos_extensions_s-G-1p25-0p25_l-U-1-20_meancs2-G-0.3-0.3_pQCD-25-40.h5`
- Place **both together** in `external/zenodo_15407795/`.

They must sit in the same directory: `eos_marginalization.py` opens the `.h5` by
a hard-coded basename relative to its own folder, and `eos_sampling/pqcd.py`
(and `run_sampling.py`, via `PQCD_CONFIG["data_path"]`) expects exactly
`external/zenodo_15407795/eos_extensions_..._pQCD-25-40.h5`. The other file in
the record, `eos_extensions_...meancs2-G-0.3-0.3.h5` (no `_pQCD-25-40`), is the
unconditioned "prior" variant and is not used by default.

---

## 4. NICER mass–radius posteriors  →  `external/zenodo_J0740/`, `external/zenodo_J0437/`, `external/zenodo_J0437_miller/`

**Only needed for the KDE / tier-2 reweighting** (`apply_kde_nicer.py`)
and the Miller robustness variant. The **baseline** `run_sampling.py` uses
summary-Gaussian NICER likelihoods with hard-coded numbers and needs **none** of
these downloads.

| Pulsar (analysis) | Zenodo DOI | Download | Extract & place at |
|-------------------|-----------|----------|--------------------|
| J0740+6620 (Salmi et al. 2024) | 10.5281/zenodo.10519473 | `mr_samples_and_contours.tar.gz` | `external/zenodo_J0740/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat` |
| J0437−4715 (Choudhury et al. 2024) | 10.5281/zenodo.13766753 | `PSR_J0437_4715_NICER_XPSI_analysis.tar.gz` (7.6 GB) | `external/zenodo_J0437/nlive20000_expf3.3_noCONST_noMM_tol0.1post_equal_weights.dat` |
| J0437−4715 (Miller et al. 2025, robustness variant only) | 10.5281/zenodo.17833896 | `J0437_NICER_RM.txt` (222 MB) | `external/zenodo_J0437_miller/J0437_NICER_RM.txt` |
| J0030+0451 (Miller et al. 2019) | — | — | Summary Gaussian, hard-coded in `eos_sampling/astro_configs.py`; no download |

Records:
- https://zenodo.org/records/10519473  (Salmi J0740)
- https://zenodo.org/records/13766753  (Choudhury J0437)
- https://zenodo.org/records/17833896  (Miller J0437)

The exact `.dat` basenames above are what `eos_sampling/astro_configs.py`
expects; the Salmi and Choudhury files live **inside** their tar archives. For
J0437, `nlive20000_..._post_equal_weights.dat` sits under
`CST_PDT/CST_PDT_outputs/3C50_BKG_AGN_smooth_3sigma_hiMN_lowXPSI_res/` inside that
7.6 GB archive; the *identical* headline M–R samples are also provided as
`J0437_3C50_BKG_AGN_hiMN_lowXPSI_mrsamples_post_equal_weights.dat` in the much
smaller `headline_result_samples_and_contours.tar.gz` (22 MB) of the **same**
record, if you prefer a lighter download. If an extracted filename differs from
what's listed, either rename it or update the `samples_path` field in
`astro_configs.py`.

---

## 5. Annala et al. 2023 comparison ensembles  →  `external/annala2023_GP/`, `external/annala2023_C4/`

**Only needed for the literature-comparison figures** (`analysis/comparisons.py`,
used by `nature_figures.py`). Loaded by explicit path, so the exact folder is up
to you — the suggestion below matches nothing hard-coded.

- **GP ensemble** — Zenodo DOI 10.5281/zenodo.10101447 ("GP ensemble of neutron
  star equations of state"). File: `EoS_ensemble.pickle`. Pass its path to
  `comparisons.load_annala_ensemble()`.
  https://zenodo.org/records/10101447
- **Interpolation (C4) ensemble** — Zenodo DOI 10.5281/zenodo.10102436
  ("Ensembles for neutron-star-matter equation-of-state interpolations").
  Provides `n_long_grid.npy` at the top level plus `main_text.zip` /
  `appendix.zip`; unzip to obtain the per-sample `.npy` arrays
  (`cs2.npy`, `mMax.npy`, `radius_MASS.npy`, `mass_grid.npy`, `Delta.npy`,
  `gamma.npy`, `dc.npy`). Point `comparisons.load_annala_interpolation_ensemble()`
  at the directory holding those `.npy` files.
  https://zenodo.org/records/10102436

---

## Resulting `external/` layout

```
external/
├── zenodo_15407795/
│   ├── eos_marginalization.py
│   └── eos_extensions_s-G-1p25-0p25_l-U-1-20_meancs2-G-0.3-0.3_pQCD-25-40.h5
├── zenodo_J0740/            # KDE workflow only
│   └── J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat
├── zenodo_J0437/            # KDE workflow only
│   └── nlive20000_expf3.3_noCONST_noMM_tol0.1post_equal_weights.dat
├── zenodo_J0437_miller/     # robustness variant only
│   └── J0437_NICER_RM.txt
├── annala2023_GP/           # comparison figures only
│   └── EoS_ensemble.pickle
└── annala2023_C4/           # comparison figures only
    ├── n_long_grid.npy
    ├── cs2.npy, gamma.npy, Delta.npy, dc.npy, mMax.npy, radius_MASS.npy, mass_grid.npy
```

`analysis/chEFT/` additionally holds `all_matter_data_high_density.csv`,
`pressure_cs2_samples.csv`, and the regenerated `*.npy` from Section 2.

---

## What each part is required for

| Task | Needs |
|------|-------|
| Regenerate χEFT anchors | §0 packages, §1 `all_matter_data_high_density.csv`, then §2 |
| Baseline sampling + reweighting (`run_sampling.py`) | §2 anchors, §3 pQCD |
| KDE NICER upgrade (`apply_kde_nicer.py`) | §3 pQCD, §4 NICER posteriors |
| Comparison figures | §5 Annala ensembles |
| PNM validation figure | §1 `pressure_cs2_samples.csv` |

## Please cite

BUQEYE nuclear-matter-convergence (Drischler et al., PRC 102, 054315 (2020);
PRL 125, 202702 (2020)); Komoltsev, Gorda & Kurkela — Zenodo 10.5281/zenodo.15407795;
Salmi et al. 2024 (J0740, Zenodo 10.5281/zenodo.10519473); Choudhury et al. 2024
(J0437, Zenodo 10.5281/zenodo.13766753); Miller et al. 2025 (J0437, Zenodo
10.5281/zenodo.17833896); Annala et al. 2023 (Zenodo 10.5281/zenodo.10101447 and
10.5281/zenodo.10102436).
