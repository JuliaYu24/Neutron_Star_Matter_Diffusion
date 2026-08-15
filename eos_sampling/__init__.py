"""
eos_sampling -- DDPM-based EOS sampler with importance reweighting.

The pipeline runs a DDPM with chi-EFT anchor points applied
as inpainting conditioning, then reweights the resulting prior samples
against astrophysical data (NICER pulsar masses/radii, gravitational-
wave tidal deformability, pulsar mass lower bound) and the
Komoltsev+2024 marginalized pQCD likelihood.

Authors: H. Alharazin, J. Yu. Panteleeva
"""


from .conditioning import build_conditioning
from .postprocess  import (denormalize_and_summarize,
                           filter_causal, summarize)
from .plotting     import (plot_results,
                           plot_nested_posteriors,
                           plot_prior_vs_posterior)
from .sampler      import sample_ddpm
from .reweighting  import importance_weights
from .pqcd         import pqcd_term, pqcd_log_likelihood
from .pipeline     import run_sampling
