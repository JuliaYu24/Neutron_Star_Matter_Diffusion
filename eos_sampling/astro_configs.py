"""
Shared NICER pulsar configurations.

Both run_sampling.py (the baseline summary-Gaussian sampling
driver) and apply_kde_nicer.py (the tier-2 KDE re-reweighting driver)
import their NICER pulsar lists from this module, so a single edit
here propagates to both workflows.

Two pre-assembled lists are exported:

  NICER_PULSARS_SUMMARY
      All three pulsars in summary-Gaussian mode.  Use this for the
      baseline run via run_sampling.py.  The headline numbers
      for J0740 and J0437 are the empirical mean +/- std of the
      published Zenodo posteriors (Salmi+24 and Choudhury+24
      respectively), which is what a Gaussian summary actually
      represents -- not the published median + asymmetric CI.

  NICER_PULSARS_KDE
      J0030 stays in summary-Gaussian mode (no public per-sample
      posterior in a single set of MR samples), J0740 and J0437
      switch to KDE-on-Zenodo-samples with explicit divide-out of
      the published priors and re-add of the radio mass measurement
      as an independent Gaussian likelihood.  Use this with
      apply_kde_nicer.py.

Switching which J0030 analysis to use (Miller+19, Riley+19,
Vinciguerra+24 ST+PDT or PDT-U) is a one-line edit to J0030_SUMMARY
below; both workflows pick it up automatically.

Switching which Zenodo file feeds the KDE for J0740 / J0437 is
a one-line edit to the "samples_path" field in J0740_KDE / J0437_KDE.

The radio-mass priors (mass_prior, mass_likelihood) only affect the
KDE workflow; the summary-Gaussian path bakes in the radio prior
implicitly through the published headline numbers.
"""

# ============================================================
#  J0030+0451 -- Miller+19 (summary in both workflows)
# ============================================================
# No public per-sample MR posterior released as a single file, so
# we keep the summary-Gaussian for both the baseline and the KDE
# workflow.  Switch to Vinciguerra+24 ST+PDT (M=1.40, sigM=0.13,
# R=11.71, sigR=0.88) by editing the four numbers below; both
# scripts will pick it up.
J0030_SUMMARY = {
    "name":    "PSR J0030+0451 (Miller+19, summary)",
    "M_obs":   1.44,   "sigma_M": 0.15,
    "R_obs":   13.02,  "sigma_R": 1.24,
}


# ============================================================
#  J0740+6620 -- Salmi+24
# ============================================================
# SUMMARY form: empirical mean and std of the Salmi+24 Zenodo
# posterior (M_samples mean=2.073, std=0.069; R_samples mean=12.652,
# std=1.081 from 379,669 samples).  Note this is the MEAN/STD of
# the posterior, not the published median + asymmetric CI -- the
# Gaussian summary represents the former, and using empirical
# moments gives the cleanest A/B against the KDE built on the same
# samples.  Previous baseline used Miller+21 (M=2.08, sigM=0.07,
# R=13.7, sigR=2.1), which is a different team's analysis on
# different data; switching to Salmi+24 here makes the
# summary-vs-KDE comparison about likelihood form, not data.
J0740_SUMMARY = {
    "name":    "PSR J0740+6620 (Salmi+24, summary)",
    "M_obs":   2.073,  "sigma_M": 0.069,
    "R_obs":   12.652, "sigma_R": 1.081,
}

# KDE form: load the Zenodo posterior samples, build a 2D KDE,
# divide out the published priors, re-add the Fonseca+21 radio
# mass measurement as an independent Gaussian likelihood.
J0740_KDE = {
    "name":             "PSR J0740+6620 (Salmi+24, KDE)",
    "mode":             "kde",
    "samples_path":     "external/zenodo_J0740/"
                        "J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat",
    "samples_columns":  (0, 1),
    "bandwidth_factor": 1.0,
    # Salmi+24 condition on the Fonseca+21 radio mass, 2.08 +/- 0.07.
    "mass_prior":       {"type": "gaussian", "M": 2.08, "sigma": 0.07},
    # X-PSI radius prior: flat in R_eq between 3 r_g(1 Msun) = 4.43 km and
    # 16 km, then modified by the compactness condition R_pol/r_g(M) > 3.
    # NOT [2GM/c^2, 16 km] -- that was never anyone's prior.
    "radius_prior":     {"type": "uniform_km", "lo": 4.43, "hi": 16.0,
                         "compactness_lo": 3.0},
    "mass_likelihood":  {"type": "gaussian", "M": 2.08, "sigma": 0.07},
}


# ============================================================
#  J0437-4715 -- Choudhury+24
# ============================================================
# SUMMARY form: empirical mean and std of the Choudhury+24 Zenodo
# posterior (M_samples mean=1.418, std=0.035; R_samples mean=11.481,
# std=0.789 from 218,553 samples).  The published headline R=11.36
# was the median; we use the empirical mean for consistency with the
# Gaussian summary, same reasoning as J0740 above.
J0437_SUMMARY = {
    "name":    "PSR J0437-4715 (Choudhury+24, summary)",
    "M_obs":   1.418,  "sigma_M": 0.035,
    "R_obs":   11.481, "sigma_R": 0.789,
}

# KDE form: same architecture as J0740_KDE.
# Radio mass prior + likelihood = Reardon+24 PPTA timing (1.418 +/- 0.044).
J0437_KDE = {
    "name":             "PSR J0437-4715 (Choudhury+24, KDE)",
    "mode":             "kde",
"samples_path":     "external/zenodo_J0437/"
                    "nlive20000_expf3.3_noCONST_noMM_tol0.1post_equal_weights.dat",
    "samples_columns":  (0, 1),       # 19-col file; M, R are columns 0, 1
    "bandwidth_factor": 1.0,
    # Choudhury+24 condition on the Reardon+24 PPTA mass, 1.418 +/- 0.044
    # (M, D and i are quoted as uncorrelated, so the marginal mass prior
    # factorises out cleanly, which is what the divide-out assumes).
    "mass_prior":       {"type": "gaussian", "M": 1.418, "sigma": 0.044},
    # Same X-PSI radius prior as J0740.
    "radius_prior":     {"type": "uniform_km", "lo": 4.43, "hi": 16.0,
                         "compactness_lo": 3.0},
    "mass_likelihood":  {"type": "gaussian", "M": 1.418, "sigma": 0.044},
}


# ============================================================
#  J0437-4715 -- Miller+25 (independent NICER analysis; robustness variant)
# ============================================================
# Second, independent NICER analysis of J0437: Miller, Dittmann, Holt,
# Lamb et al. 2026 (ApJL 1000, L48; arXiv:2512.08790).  Favored-model
# M-R posterior on Zenodo 10.5281/zenodo.17833896 (J0437_NICER_RM.txt:
# col1=R, col2=M, col3=weight).  That file is WEIGHTED, unlike the
# equal-weight Choudhury file, so it must first be resampled to equal
# weight (weight column dropped) before the KDE loader -- which ignores
# weights -- reads it.  The path/columns below assume the resampled
# equal-weight file saved as (M, R); adjust if you saved it differently.
#
# NOTE: this posterior is BIMODAL.  The KDE captures both modes; if
# Scott's-rule bandwidth visibly merges them, drop bandwidth_factor to
# ~0.6-0.8 and recheck.  Priors carry over from J0437_KDE: same Reardon+24
# radio mass (mass_prior == mass_likelihood cancels), uniform radius prior
# (a constant in the relative weights), so the values below need not match
# Miller's ranges exactly.
J0437_KDE_MILLER = {
    "name":             "PSR J0437-4715 (Miller+25, KDE)",
    "mode":             "kde",
    # Point straight at the raw Zenodo deposit: (R, M, weight), so the
    # (M, R) column order is (1, 0) and the weights are column 2.  The
    # KDE consumes the weights natively -- no hand-made equal-weight
    # file, so nothing derived has to be tracked or regenerated.
    "samples_path":     "external/zenodo_J0437_miller/J0437_NICER_RM.txt",
    "samples_columns":  (1, 0),       # file is (R, M); we want (M, R)
    "weights_column":   2,
    "bandwidth_factor": 1.0,
    # Miller+25 Table 1: same Reardon+24 radio mass prior as Choudhury+24,
    # so mass_prior == mass_likelihood cancels here too.
    "mass_prior":       {"type": "gaussian", "M": 1.418, "sigma": 0.044},
    # DIFFERENT radius prior from the Amsterdam analyses: Miller+25 sample
    # flat in the INVERSE COMPACTNESS c^2 R_e/(GM) over [3.2, 8.0], so at
    # fixed M the radius is flat over [3.2 r_g(M), 8.0 r_g(M)] and the
    # conditional normalisation goes as 1/M rather than being constant.
    # (Their Fig. 5 confirms the upper M-R boundary is the c^2R/(GM)=8
    # prior edge, ~16.7 km at 1.418 Msun -- above the 16 km ceiling that
    # the Amsterdam runs impose.)
    "radius_prior":     {"type": "uniform_compactness",
                         "lo_over_rg": 3.2, "hi_over_rg": 8.0},
    "mass_likelihood":  {"type": "gaussian", "M": 1.418, "sigma": 0.044},
}


# ============================================================
#  Assembled lists for direct import
# ============================================================
NICER_PULSARS_SUMMARY   = [J0030_SUMMARY, J0740_SUMMARY, J0437_SUMMARY]
NICER_PULSARS_KDE       = [J0030_SUMMARY, J0740_KDE,     J0437_KDE]
NICER_PULSARS_KDE_MILLER = [J0030_SUMMARY, J0740_KDE,     J0437_KDE_MILLER]


__all__ = [
    "J0030_SUMMARY",
    "J0740_SUMMARY", "J0740_KDE",
    "J0437_SUMMARY", "J0437_KDE", "J0437_KDE_MILLER",
    "NICER_PULSARS_SUMMARY",
    "NICER_PULSARS_KDE",
    "NICER_PULSARS_KDE_MILLER",
]
