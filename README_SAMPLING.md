# Sampling — posterior generation and reweighting

The sampling stage draws equation-of-state samples from the trained diffusion
prior and conditions them on astrophysical + pQCD data by **post-hoc importance
reweighting**. Three scripts, run in sequence; only the first draws samples and
solves TOV — the other two are seconds of NumPy on cached arrays.

| Script | Does | Draws / TOV? |
|--------|------|--------------|
| `run_sampling.py` | Draws \(N\) samples from the DDPM prior (χEFT anchors inpainted), TOV-solves each, and applies the **summary-mode** NICER + GW + \(M_\mathrm{max}\) + pQCD reweighting. Writes a `.pt`. | yes |
| `apply_kde_nicer.py` | Re-reweights a saved `.pt`, upgrading the NICER term to the **tier-2 KDE** likelihood (J0740 Salmi + J0437 Choudhury *or* Miller; J0030 stays summary). Reuses cached `M`/`R`/`Lambda` and the GW/\(M_\mathrm{max}\)/pQCD log-L. | no |
| `apply_heavy_mass_constraint.py` | Multiplies the existing weights by a one-sided heavy-mass (\(M_\mathrm{TOV}\) floor) likelihood on the cached \(M_\mathrm{max}\) (default PSR J0952−0607, \(2.35\pm0.17\,M_\odot\)). | no |

Flow: `run_sampling.py` → summary `.pt` → `apply_kde_nicer.py` → KDE `.pt` →
*(optional)* `apply_heavy_mass_constraint.py` → `+J0952` `.pt`. Each script has an
input/output path at the top of its config block; point the analysis notebook's
`OUTPUT_PATH` at whichever `.pt` you want to inspect.

## Where each variation is set

- **Sample count** — `n_samples` in `run_sampling.py` (100 000 or 25 000).
- **χEFT cutoff Λ** — `LAMBDA` in `run_sampling.py` (500 or 450); selects the
  `cs2_BETAEQ_Lambda-<Λ>_*` anchors from `analysis/chEFT/`.
- **Prior / model** — `checkpoint_path` in `run_sampling.py`; the "+class"
  robustness runs use a model retrained with one extra validation class added
  (class 11 = `gp_cs2`, 12 = `pwlinear_cs2`, 13 = `metamodel`, from
  `eos_class_validation_curves/`).
- **J0437 NICER analysis** — `J0437_ANALYSIS` in `apply_kde_nicer.py`
  (`"choudhury"` or `"miller"`; `"summary"` reproduces the tier-1 term).
- **Heavy mass** — `M_LOW` / `SIGMA` / `ONE_SIDED` in
  `apply_heavy_mass_constraint.py` (default J0952, one-sided floor).

The baseline always carries the GW170817 tidal term and an \(M_\mathrm{max}\)
floor at \(1.908\,M_\odot\) (J1614−2230); run 2 adds J0952 on top of that.

## Production runs

Every run does summary reweighting first (`run_sampling.py`), then the KDE
(`apply_kde_nicer.py`, Choudhury unless noted).

| # | Model · N | Λ | KDE | Extra step | Purpose |
|---|-----------|---|-----------|------------|---------|
| 1 | baseline · 100 000 | 500 | Choudhury | — | **final result** |
| 2 | baseline · 100 000 | 500 | Choudhury | `apply_heavy_mass_constraint.py` on run 1 (+J0952) | effect of a heavy-mass constraint |
| 3 | baseline · 100 000 | 500 | **Miller** | — | effect of Miller's larger error bars |
| 4 | baseline · 25 000 | 500 | Choudhury | — | robustness — sample count |
| 5 | baseline · 25 000 | **450** | Choudhury | — | robustness — χEFT cutoff |
| 6 | baseline **+ class 11** · 25 000 | 500 | Choudhury | — | robustness — prior (+`gp_cs2`) |
| 7 | baseline **+ class 12** · 25 000 | 500 | Choudhury | — | robustness — prior (+`pwlinear_cs2`) |
| 8 | baseline **+ class 13** · 25 000 | 500 | Choudhury | — | robustness — prior (+`metamodel`) |

Run 1 is the headline posterior; run 2 reweights it; run 3 swaps only the J0437
likelihood; runs 4–8 are independent 25 000-sample re-runs that vary one
ingredient each.

## Inputs & outputs

`run_sampling.py` needs the trained checkpoint (`checkpoints/eos_ddpm_best.pt`),
the χEFT anchors in `analysis/chEFT/` (`README_EXTERNAL_DATA.md` §1–2) and the
pQCD data (`README_EXTERNAL_DATA.md` §3). The KDE step (`apply_kde_nicer.py`) additionally needs the NICER posteriors (§4). Each script writes a `.pt` and never edits its input;
`apply_*` also stash traceability fields (`weights_preJ`, `heavy_mass_constraint`,
`log_L_nicer_kde`, …) alongside the new weights.
