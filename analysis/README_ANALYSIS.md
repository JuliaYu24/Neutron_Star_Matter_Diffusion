# Analysis — derived quantities, diagnostics, and figures

The `analysis/` package turns a sampling posterior into the paper's derived
quantities and figures. It is pure post-processing of the `.pt` written by
`run_sampling.py`: it reuses the Heun integrator, the TOV stable-branch logic and
the weighted-quantile utility from `eos_sampling.reweighting`, and adds
pressure / energy-density reconstruction, the conformality diagnostics
(\(\gamma, \Delta, d_c\)), fiducial NS quantities, a phase-transition flag, the
symmetry-energy slope \(L\), literature comparisons, and the publication figures.
No new sampling, TOV solving or pQCD evaluation happens here.

`notebook_diagnostics_kde.ipynb` is the driver; the `.py` modules are the library
it calls (all re-exported from the package root).

## Layout

| Path | Role |
|------|------|
| `notebook_diagnostics_kde.ipynb` | Driver — loads the posterior, calls everything, writes the figures. Start here. |
| `diagnostics.py` | Derived quantities: \(P\), \(\varepsilon\), \(\gamma\), \(\Delta\), \(d_c\), fiducial NS quantities, phase-transition flag, \(L\). |
| `plots.py` | Figure primitives (diagnostics panels, \(P(\varepsilon)\) and \(M\)–\(R\) bands, legends, training curves). |
| `comparisons.py` | Loads the Annala et al. 2023 GP and C4 ensembles and reduces them to overlay bands. |
| `nature_figures.py` | Assembles the combined publication figures from those primitives. |
| `chEFT/` | χEFT anchor extraction — see `README_EXTERNAL_DATA.md` §1–2. |

## Input

Everything keys off the dict written by `run_sampling.py` (loaded via
`torch.load`): `samples_phys` (per-sample \(c_s^2(n_B)\)), `nB_grid`, the
`reweighting` block (importance `weights`, cached TOV `M`/`R`/`Lambda`, ESS),
`ref_point` (`eps_ref`, `P_ref` from the χEFT anchor) and `cs2_cov`/`cs2_sigma`.
The shipped notebook uses `res_finale_baseline/eos_EFT_posterior_kdenicer.pt`; set
`OUTPUT_PATH` to your own.

## Quick start

From the `analysis/` folder, open the notebook, set `OUTPUT_PATH` (§2), make sure
the external inputs below are reachable, and run top to bottom. Figures are
written to `plots/`. Its 16 sections mirror the module API: \(P,\varepsilon\) and
the conformality trio, the phase-transition posterior, fiducial NS quantities, the
Annala GP/C4 comparisons, the combined Nature figures, and the fiducials table and
forest plot.

## Modules

- **`diagnostics.py`** — `compute_P_eps_ensemble` (Heun integration to \(P,
  \varepsilon\)), `compute_diagnostics` (\(\gamma, \Delta, d_c\)),
  `fiducial_NS_quantities`, `phase_transition_diagnostic`, and three
  symmetry-energy-slope estimators (`compute_L_per_sample_textmethod`,
  `compute_L_per_sample_marginalized`, `compute_L_from_S2_samples`) with their
  BUQEYE SNM baselines. Reads the χEFT `.npy` (default `chEFT/`, `Lambda=500`);
  writes nothing.
- **`plots.py`** — the visualization layer: `plot_diagnostics_4panel`,
  `plot_P_eps_band`, `plot_M_R_band` (numbered NICER markers),
  `plot_pt_peak_distribution`, the shared-legend helpers and
  `plot_training_curves`. Functions return `(fig, axes)` and save only to a
  caller-supplied path; the palette matches `eos_sampling.plotting`.
- **`comparisons.py`** — `load_annala_ensemble` (GP pickle) and
  `load_annala_interpolation_ensemble` (C4 directory of `.npy`), plus
  `compute_external_band` / `compute_external_M_R_band` to turn either into overlay
  bands. The C4 reconstruction needs an \(\varepsilon, P\) anchor at ~1 \(n_0\),
  printed by `chEFT/extract_anchor_1n0.py` (paste into the notebook's `C4_ANCHOR`).
- **`nature_figures.py`** — `figure_diagnostics_2x2`, `figure_mr_pfree`,
  `figure_fiducials_forest`: a library (no `__main__`) that arranges the primitives
  into the paper's multi-panel figures. Needs the shipped patched `plots.py`
  (panels accept `ax=None`).

## Annala inputs

What the two loaders actually read:

- **GP** (`EoS_ensemble.pickle`, Zenodo 10101447) — a pickled `(df, n)`; from `df`
  the loader takes the columns `cs2`, `e`, `p` (GeV/fm³), the published weight
  columns `X_rays`, `r_J0348`, `QCD_10ns`, `TD_BH`, the TOV arrays `m`, `r`, `L`,
  and `mmax`.
- **C4** (`.npy` directory, Zenodo 10102436) — required `cs2.npy` and
  `n_long_grid.npy`; optional `mMax.npy`, `radius_MASS.npy`, `mass_grid.npy`,
  `Delta.npy`, `gamma.npy`, `dc.npy`; uniform weights.

## External inputs

| Input | For | Source |
|-------|-----|--------|
| posterior `.pt` | everything | `run_sampling.py` |
| `EoS_ensemble.pickle` | Annala GP overlay | Zenodo 10101447 (`README_EXTERNAL_DATA.md` §5) |
| `C4/` `.npy` | Annala C4 overlay | Zenodo 10102436 (`README_EXTERNAL_DATA.md` §5) |
| `chEFT/` `.npy` | SNM baseline and \(L\) | regenerate with `chEFT/cs2_betaeq_anchors.py` (§2) |

## Outputs

Figures under `plots/`: `training_curves.png`, `eos_EFT_posterior_nested.png`,
`fig_diagnostics_nature.png`, `fig_mr_pfree_nature.png`,
`fiducials_forest_nature.png`. The master fiducials table (§15) is shown inline in
the notebook, not saved.

## Dependencies

`numpy`, `scipy`, `torch`, `matplotlib` (the notebook also uses `pandas` and
`IPython`); internally `eos_sampling.reweighting` and `eos_sampling.plotting`.
