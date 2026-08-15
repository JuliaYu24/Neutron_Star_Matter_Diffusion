"""
EOS sampling pipeline: DDPM prior + SciPy importance reweighting.

Astrophysical conditioning (NICER, GW, M_max) enters as an
importance-weighting pass over the DDPM samples.  Optionally, a Komoltsev+2024
marginalized pQCD likelihood is also applied in the same reweighting
pass (see eos_sampling.pqcd).

Pipeline:
  load_model
    -> build_conditioning (chi-EFT anchors only; optional cs2_cov)
    -> sample_ddpm (replacement inpainting + anchor jitter)
    -> denormalize_and_summarize
    -> (optional) filter_causal
    -> importance_weights (SciPy adaptive RK45, parallelized, + pQCD)
    -> save .pt

NO PLOTTING IS DONE HERE.  The .pt file produced by this orchestrator
contains everything needed for plotting and analysis.
Open notebook_diagnostics_kde.ipynb to render the EOS reconstruction figure
plus the nested-posterior and prior-vs-posterior bands (via
eos_sampling.plotting).
"""

import os
import time
import torch

from eos_diffusion.inference import load_model

from .conditioning  import build_conditioning
from .sampler       import sample_ddpm
from .postprocess   import denormalize_and_summarize, filter_causal, summarize
from .reweighting   import importance_weights, weighted_summary


def run_sampling(nB_known, cs2_known, cs2_sigma=None, cs2_cov=None,
                 checkpoint_path="checkpoints/eos_ddpm_best.pt",
                 n_samples=200, batch_size=20, seed=None,
                 enforce_causality=True, cs2_bounds=(0.0, 1.0),
                 astro_config=None, reweighting_config=None,
                 pqcd_config=None, reuse_raw=False,
                 save_prefix="eos_EFT_posterior"):
    """
    Run the EOS sampling pipeline end-to-end.

    Required
    --------
    nB_known, cs2_known : (K,) anchor points
    astro_config        : dict with nicer_pulsars / gw / mmax (+ TOV
                          numerics); used by the SciPy reweighter.

    Optional
    --------
    cs2_sigma           : (K,) independent errors, or None
    cs2_cov             : (K, K) correlated covariance, or None.
                          If given, overrides cs2_sigma.
    reweighting_config  : dict (rtol, atol, verbose, n_jobs, ...)
    pqcd_config         : dict or None.  Komoltsev+2024 marginalized
                          pQCD config (see eos_sampling.pqcd).
                          None disables pQCD entirely.

    Individual astrophysical conditioning terms can also be disabled
    by removing them from astro_config:
      * NICER off  : empty/missing "nicer_pulsars" list
      * GW    off  : "gw": None (or missing)
      * M_max off  : "mmax": None (or missing)

    Returns
    -------
    output : dict of sample/summary tensors, plus
        reweighting     : dict from importance_weights including M, R,
                          Lambda arrays (cached for cheap re-reweighting)
                          and per-term ESS.  When pqcd_config is active,
                          also contains log_L_pqcd, pqcd_passed,
                          pqcd_n_T, pqcd_mu_T, ESS_pqcd.
        weighted_mean   : (L,) posterior mean
        weighted_std    : (L,) posterior std
        weighted_q16    : (L,) 16th posterior quantile
        weighted_q84    : (L,) 84th posterior quantile
        n_drawn         : N before causality filter
        n_post_filter   : N after causality filter
    """
    print("=" * 60)
    print("  EOS Sampling pipeline")
    print("=" * 60)

    # Make sure the output directory exists before any torch.save fires.
    # save_prefix may be a bare filename ('foo') or include a folder
    # ('analysis/foo'); only mkdir when there is a folder component.
    out_dir = os.path.dirname(save_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Output directory: {out_dir}/")

    model, schedule, config, norm, nB_grid = load_model(checkpoint_path)
    device = next(model.parameters()).device

    if norm is None:
        raise RuntimeError(
            "Checkpoint does not contain normalization stats ('mean', 'std').")
    norm_mean = norm['mean'].float().cpu()
    norm_std  = norm['std'].float().cpu()

    if nB_grid is None:
        raise RuntimeError("Checkpoint does not contain 'nB_over_n0_grid'.")
    nB_grid = nB_grid.float().cpu()
    print(f"  nB/n0 grid: {tuple(nB_grid.shape)} "
          f"in [{nB_grid.min():.3f}, {nB_grid.max():.3f}]")

    print("\nBuilding inpainting conditioning (chi-EFT anchors)...")
    mask, x_cond, sigma_cond, indices, anchor_chol = build_conditioning(
        nB_grid, nB_known, cs2_known, norm_mean, norm_std,
        cs2_sigma=cs2_sigma, cs2_cov=cs2_cov)

    anchor_indices_tensor = (torch.tensor(indices, dtype=torch.long)
                             if anchor_chol is not None else None)

    if astro_config is None:
        raise ValueError(
            "run_sampling requires astro_config for the reweighter; pass "
            "a dict with at least one of nicer_pulsars / gw / mmax.")

    if (astro_config.get("eps_ref") is None
            or astro_config.get("P_ref") is None):
        raise ValueError(
            "astro_config must carry explicit 'eps_ref' and 'P_ref' "
            "(MeV/fm^3 at the first grid point).  Load them from the "
            "chEFT reference-point .npz before calling run_sampling -- "
            "failing here, before any sampling cost is paid, on purpose.")
    _ref_src = astro_config.get("ref_source", "")
    print(f"  Reference point: eps_ref = {astro_config['eps_ref']:.4f}, "
          f"P_ref = {astro_config['P_ref']:.4f} MeV/fm^3"
          + (f"   [{_ref_src}]" if _ref_src else ""))


    raw_path = f"{save_prefix}_raw.pt"
    if reuse_raw and os.path.exists(raw_path):
        cached = torch.load(raw_path, map_location="cpu", weights_only=False)
        # The cache holds NORMALISED draws; denormalising them with a
        # different checkpoint's (mean, std) produces an ensemble that
        # was never sampled from anything, so the cache identity is
        # checked in full before a single curve is denormalised.
        _checks = (
            ("mask",       mask,       cached.get("mask")),
            ("x_cond",     x_cond,     cached.get("x_cond")),
            ("cs2_known",  cs2_known,  cached.get("cs2_known")),
            ("nB_known",   nB_known,   cached.get("nB_known")),
            ("cs2_cov",    cs2_cov,    cached.get("cs2_cov")),
            ("cs2_sigma",  cs2_sigma,  cached.get("cs2_sigma")),
            ("norm_mean",  norm_mean,  cached.get("norm_mean")),
            ("norm_std",   norm_std,   cached.get("norm_std")),
            ("nB_grid",    nB_grid,    cached.get("nB_grid")),
        )
        unverifiable = []
        for name, current, old in _checks:
            if old is None and current is not None and "norm_mean" not in cached:
                unverifiable.append(name)
                continue
            if current is None and old is None:
                continue
            if (current is None) != (old is None):
                raise SystemExit(
                    f"{raw_path} was generated with {name}="
                    f"{'None' if old is None else 'set'} but this run has "
                    f"{name}={'None' if current is None else 'set'}.  Delete "
                    f"the cache or change save_prefix.")
            cur_t = torch.as_tensor(current).float()
            old_t = torch.as_tensor(old).float()
            if cur_t.shape != old_t.shape or not torch.allclose(
                    cur_t, old_t, rtol=1e-5, atol=1e-7):
                raise SystemExit(
                    f"{raw_path} was generated with a different {name}; the "
                    f"cached draws do not belong to this configuration.  "
                    f"Delete the cache or change save_prefix.\n"
                    f"  (checkpoint recorded in the cache: "
                    f"{cached.get('checkpoint_path', 'not recorded')})")

        if "checkpoint_path" not in cached:
            unverifiable.append("checkpoint_path")
        elif cached["checkpoint_path"] != checkpoint_path:
            raise SystemExit(
                f"{raw_path} was generated from checkpoint "
                f"{cached['checkpoint_path']!r}, this run uses "
                f"{checkpoint_path!r}.  Delete the cache or change "
                f"save_prefix.")

        if unverifiable:
            if reuse_raw != "unverified":
                raise SystemExit(
                    f"{raw_path} predates the cache fingerprint: "
                    f"{', '.join(sorted(set(unverifiable)))} are not recorded "
                    f"in it, so the cached draws cannot be checked against "
                    f"this checkpoint's normalisation.\n"
                    f"  * If the checkpoint has NOT changed since that file "
                    f"was written, set reuse_raw=\"unverified\" to accept it.\n"
                    f"  * If it has, or you are unsure, delete {raw_path} and "
                    f"re-sample: reusing draws across a checkpoint change "
                    f"produces an ensemble that was never sampled from "
                    f"anything.\n"
                    f"  The _raw.pt this run writes carries the fingerprint, "
                    f"so this is a one-time decision.")
            print(f"  WARNING: accepting an UNVERIFIED cache "
                  f"({', '.join(sorted(set(unverifiable)))} not recorded); "
                  f"you have asserted the checkpoint is unchanged since "
                  f"{raw_path} was written.")

        samples_norm = cached['samples_norm']
        if samples_norm.shape[0] < n_samples:
            raise SystemExit(
                f"{raw_path} holds {samples_norm.shape[0]} samples but "
                f"n_samples={n_samples} was requested.  Silently running on "
                f"the smaller ensemble is how a cache stops matching its "
                f"config; delete it or lower n_samples.")
        if samples_norm.shape[0] > n_samples:
            print(f"  Trimming cache {samples_norm.shape[0]} -> {n_samples}")
            samples_norm = samples_norm[:n_samples]
        print(f"\nReusing {raw_path}  {tuple(samples_norm.shape)}")
        if unverifiable:
            print(f"  (checked: mask, x_cond, anchors, covariance, grid.  "
                  f"NOT checked: {', '.join(sorted(set(unverifiable)))})")
        else:
            print(f"  (cache identity verified against mask, x_cond, anchors, "
                  f"covariance, grid and checkpoint normalisation)")
    else:
        print(f"\nSampling {n_samples} curves (T={schedule.T} steps, "
              f"B={batch_size})...")
        t_start = time.time()
        samples_norm = sample_ddpm(
            model, schedule, mask, x_cond,
            sigma_cond=sigma_cond,
            anchor_chol=anchor_chol,
            anchor_indices=anchor_indices_tensor,
            n_samples=n_samples, batch_size=batch_size,
            device=device, seed=seed)
        elapsed = time.time() - t_start
        print(f"  Output shape: {tuple(samples_norm.shape)}")
        print(f"  Time: {elapsed:.1f}s  ({elapsed / n_samples:.2f}s/sample)")

        torch.save({
            'samples_norm': samples_norm,
            'nB_grid':      nB_grid,
            'nB_known':     nB_known,
            'cs2_known':    cs2_known,
            'cs2_sigma':    cs2_sigma,
            'cs2_cov':      cs2_cov,
            'mask':         mask,
            'x_cond':       x_cond,
            'norm_mean':       norm_mean,
            'norm_std':        norm_std,
            'checkpoint_path': checkpoint_path,
            'n_samples':       int(n_samples),
            'seed':            seed}, raw_path)
        print(f"  Raw normalized samples saved to {raw_path}")


    print("\nDe-normalizing and computing statistics...")
    samples_phys, mean_curve, std_curve, median_curve, q16, q84 = \
        denormalize_and_summarize(samples_norm, norm_mean, norm_std)

    n_drawn = samples_phys.shape[0]

    if enforce_causality:
        samples_phys = filter_causal(samples_phys, *cs2_bounds)
        if samples_phys.shape[0] == 0:
            raise RuntimeError("All samples filtered out by causality cut.")
        mean_curve, std_curve, median_curve, q16, q84 = summarize(samples_phys)

    n_post_filter = samples_phys.shape[0]

    print("\nFidelity check at known points (unweighted):")
    print(f"  {'Idx':>5s}  {'nB/n0':>8s}  {'True':>10s}  {'Mean':>10s}  "
          f"{'|Delta|':>10s}  {'Rel %':>7s}")
    print(f"  {'-' * 60}")
    for k, j in enumerate(indices):
        true_val = cs2_known[k].item()
        pred_val = mean_curve[j].item()
        delta    = abs(pred_val - true_val)
        rel_pct  = 100 * delta / max(abs(true_val), 1e-10)
        print(f"  {j:5d}  {nB_known[k]:8.3f}  "
              f"{true_val:10.4f}  {pred_val:10.4f}  "
              f"{delta:10.4f}  {rel_pct:6.1f}%")

    rc = reweighting_config if reweighting_config is not None else {}
    print("\nRunning SciPy importance reweighting...")
    pqcd_on = pqcd_config is not None
    if pqcd_on:
        print(f"  pQCD likelihood: Komoltsev+2024 marginalized, "
              f"n_T={pqcd_config.get('n_T_over_n0', 'n_TOV')}")
    reweighting = importance_weights(
        samples_phys    = samples_phys,
        nB_grid         = nB_grid,
        guidance_config = astro_config,
        nicer_mode      = rc.get("nicer_mode",
                           astro_config.get("nicer_mode", "line_integral")),
        rtol            = rc.get("rtol", 1e-6),
        atol            = rc.get("atol", 1e-8),
        verbose         = rc.get("verbose", True),
        n_jobs          = rc.get("n_jobs", -1),
        pqcd_config     = pqcd_config)

    # weighted_summary returns (mean, std, q16, q84); the std is what
    # notebook_diagnostics_kde.ipynb plots as the "+/-1 sigma" band.
    weighted_mean, weighted_std, weighted_q16, weighted_q84 = weighted_summary(
        samples_phys.numpy(), reweighting["weights"])

    # Report ESS as a fraction of both N_post_filter (intrinsic posterior
    # variance) and N_drawn (total sampling-effort efficiency).
    ess       = reweighting["ESS"]
    ess_pct_f = 100.0 * ess / n_post_filter
    ess_pct_d = 100.0 * ess / n_drawn
    print(f"\n  ==========================================================")
    print(f"  ESS (joint) = {ess:.1f}")
    print(f"     / N_post_filter = {n_post_filter}  ->  {ess_pct_f:.1f}%   "
          f"(posterior variance scale)")
    print(f"     / N_drawn       = {n_drawn}  ->  {ess_pct_d:.1f}%   "
          f"(sampling-effort efficiency, includes causality-filtered draws)")
    if pqcd_on:
        n_pass = int(reweighting["pqcd_passed"].sum())
        print(f"  pQCD pass rate  = {n_pass}/{reweighting['N']}  "
              f"({100.0 * n_pass / reweighting['N']:.1f}%)")
    print(f"  ==========================================================")
    if ess_pct_d < 1.0:
        print("  WARNING: ESS/N_drawn < 1%.  The prior rarely")
        print("  produces samples consistent with the astrophysical data.")
        print("  Consider drawing more samples or revisiting the anchor")
        print("  configuration.")
    elif ess_pct_d < 5.0:
        print("  NOTE: ESS/N_drawn < 5%.  Posterior is valid but sample")
        print("  efficiency is modest; consider increasing n_samples.")

    print("\nOpen notebook_diagnostics_kde.ipynb on the saved .pt to "
          "render figures.")

    output = {
        'nB_grid':       nB_grid,
        'samples_phys':  samples_phys,
        'mean_curve':    mean_curve,
        'std_curve':     std_curve,
        'median_curve':  median_curve,
        'q16':           q16,
        'q84':           q84,
        'nB_known':      nB_known,
        'cs2_known':     cs2_known,
        'cs2_sigma':     cs2_sigma,
        'cs2_cov':       cs2_cov,
        'mask':          mask,
        'x_cond':        x_cond,
        'reweighting':   reweighting,
        'weighted_mean': weighted_mean,
        'weighted_std':  weighted_std,
        'weighted_q16':  weighted_q16,
        'weighted_q84':  weighted_q84,
        'n_drawn':       n_drawn,
        'n_post_filter': n_post_filter,
        'ref_point': {
            'eps_ref': float(astro_config['eps_ref']),
            'P_ref':   float(astro_config['P_ref']),
            'source':  astro_config.get('ref_source', ''),
        },
    }

    out_path = f"{save_prefix}.pt"
    torch.save(output, out_path)
    print(f"  Results saved to {out_path}")
    print(f"  (output['reweighting'] now carries M, R, Lambda arrays of")
    print(f"   shape (N, n_central) for later re-reweighting.)")
    print("\nDone.\n")
    return output