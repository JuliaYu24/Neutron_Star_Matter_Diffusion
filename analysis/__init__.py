"""
analysis -- post-hoc derived quantities from EOS sampling posteriors.

Operates on the dict produced by eos_sampling.run_sampling (loaded from
a .pt or held in memory).  Reuses the validated Heun integrator, the
TOV stable-branch logic, and the weighted-quantile utility from
eos_sampling.reweighting; adds new derived quantities (P, eps, gamma,
Delta, d_c), the symmetry-energy slope L, fiducial-mass NS quantities,
a phase-transition flag, and matching plots.

Authors: J. Yu. Panteleeva, H. Alharazin
"""

from .diagnostics import (
    compute_P_eps_ensemble,
    compute_diagnostics,
    fiducial_NS_quantities,
    print_fiducial_table,
    phase_transition_diagnostic,
    symmetric_matter_baseline,
    symmetric_matter_baseline_draws,
    compute_L_per_sample_textmethod,
    compute_L_per_sample_marginalized,
    compute_L_from_S2_samples,
)

from .plots import (
    plot_diagnostics_4panel,
    plot_P_eps_band,
    plot_M_R_band,
    merge_pulsar_legend,
    plot_pt_peak_distribution,
    make_shared_diagnostics_legend,
    add_shared_diagnostics_legend,
)

from .comparisons import (
    load_annala_ensemble,
    load_annala_interpolation_ensemble,
    compute_external_band,
    compute_external_M_R_band,
)
