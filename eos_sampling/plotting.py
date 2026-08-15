"""
Plotting: EOS reconstruction, nested posterior bands, prior-vs-posterior.

All functions in this module are pure visualization on the dict produced
by run_sampling (loaded from a .pt or held in memory).  No sampling,
no TOV, no pQCD recomputation -- the dict already carries every array
the figures need.
"""

import numpy as np
import matplotlib.pyplot as plt

from eos_sampling.reweighting import weighted_quantile

# ================================================================
# Internal helpers
# ================================================================
def _to_np(x):
    """Tolerant tensor->numpy conversion."""
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)


def _draw_causal_region(ax, *, draw_band=True, band_alpha=0.05,
                        band_label=None, edges=True, edge_lw=0.8):
    """
    Shared background for every panel: the causal region 0 <= c_s^2 <= 1
    (faint gray fill, optional dotted edges) plus the conformal-limit
    reference line at c_s^2 = 1/3.  The conformal line is always drawn.
    """
    if draw_band:
        ax.axhspan(0.0, 1.0, color="gray", alpha=band_alpha, label=band_label)
        if edges:
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=edge_lw)
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=edge_lw)
    ax.axhline(1.0 / 3.0, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5, zorder=1)


def _softmax_weights(log_w):
    """Stable softmax + ESS.  Mirrors reweighting._ess_from_logw."""
    log_w  = np.asarray(log_w, dtype=np.float64)
    N      = log_w.shape[0]
    finite = np.isfinite(log_w)
    if not finite.any():
        return np.full(N, 1.0 / N), float(N)
    m       = log_w[finite].max()
    shifted = np.where(finite, log_w - m, -np.inf)
    raw     = np.exp(shifted)
    Z       = raw.sum()
    if Z <= 0:
        return np.full(N, 1.0 / N), float(N)
    w   = raw / Z
    ess = 1.0 / np.sum(w * w)
    return w, float(ess)


def _weighted_quantile(samples, weights, q):
    """
    Pointwise weighted quantile across the sample axis.

    samples : (N, L) array
    weights : (N,)  non-negative array (need not be normalized)
    q       : scalar in [0, 1]

    Returns (L,) array of weighted q-quantiles, one per grid point.
    """
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0 or not np.isfinite(weights.sum()):
        return np.quantile(samples, q, axis=0)
    return np.array([weighted_quantile(samples[:, j], weights, q)
                     for j in range(samples.shape[1])])


def _anchor_err_from_output(output):
    """
    Return a (K,) numpy array of 1-sigma anchor errors derived from
    whichever uncertainty representation the run used.

    Priority: cs2_cov  -> sqrt(diag(cov))
              cs2_sigma -> as-is
              neither   -> None  (caller falls back to plain scatter)
    """
    cov = output.get("cs2_cov")
    if cov is not None:
        cov = _to_np(cov)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    sig = output.get("cs2_sigma")
    if sig is not None:
        return _to_np(sig)
    return None


def _nested_posterior_bands(output, q_low=0.16, q_high=0.84):
    """
    Build three nested c_s^2 bands showing how each successive layer
    of conditioning narrows the EOS reconstruction:

        1. chi_eft             : chi-EFT inpainting only (no reweighting)
        2. chi_eft_astro       : chi-EFT + NICER + GW + M_max
        3. chi_eft_astro_pqcd  : chi-EFT + astro + pQCD  (full posterior)
        
    """
    rw      = output["reweighting"]
    samples = _to_np(output["samples_phys"]).astype(np.float64)
    N, _    = samples.shape

    bands = {}

    # ---- 1. chi-EFT only (inpainted, unweighted) ----
    bands["chi_eft"] = {
        "median": np.quantile(samples, 0.50,  axis=0),
        "q_low":  np.quantile(samples, q_low, axis=0),
        "q_high": np.quantile(samples, q_high, axis=0),
        "n":      N,
        "ess":    float(N),
        "label":  r"$\chi$EFT inpainting"}

    # ---- 2. chi-EFT + astrophysical data ----
    log_L_astro = np.zeros(N)
    astro_terms = []
    for k, n in [("log_L_nicer", "NICER"),
                 ("log_L_gw",    "GW"),
                 ("log_L_mmax",  r"$M_\mathrm{max}$")]:
        if k in rw:
            log_L_astro = log_L_astro + np.asarray(rw[k], dtype=np.float64)
            astro_terms.append(n)

    if astro_terms:
        w_astro, ess_astro = _softmax_weights(log_L_astro)
        bands["chi_eft_astro"] = {
            "median": _weighted_quantile(samples, w_astro, 0.50),
            "q_low":  _weighted_quantile(samples, w_astro, q_low),
            "q_high": _weighted_quantile(samples, w_astro, q_high),
            "n":      N,
            "ess":    ess_astro,
            "label":  r"$\chi$EFT $+$ astro ("
                      + " $+$ ".join(astro_terms) + ")"}

    # ---- 2b. chi-EFT + pQCD only (isolates the pQCD effect) ----
    if "log_L_pqcd" in rw:
        w_pqcd, ess_pqcd = _softmax_weights(
            np.asarray(rw["log_L_pqcd"], dtype=np.float64))
        bands["chi_eft_pqcd"] = {
            "median": _weighted_quantile(samples, w_pqcd, 0.50),
            "q_low":  _weighted_quantile(samples, w_pqcd, q_low),
            "q_high": _weighted_quantile(samples, w_pqcd, q_high),
            "n":      N,
            "ess":    ess_pqcd,
            "label":  r"$\chi$EFT $+$ pQCD"}

    # ---- 3. chi-EFT + astro + pQCD (= full posterior) ----
    if "log_L_pqcd" in rw:
        # Use the canonical 'weights' from the reweighter so this
        # matches output['weighted_*'] exactly.
        w_full = np.asarray(rw["weights"], dtype=np.float64)
        bands["chi_eft_astro_pqcd"] = {
            "median": _weighted_quantile(samples, w_full, 0.50),
            "q_low":  _weighted_quantile(samples, w_full, q_low),
            "q_high": _weighted_quantile(samples, w_full, q_high),
            "n":      N,
            "ess":    float(rw["ESS"]),
            "label":  r"$\chi$EFT $+$ astro $+$ pQCD (full posterior)"}

    return bands


# ================================================================
# Plot 1: EOS reconstruction
# ================================================================
def plot_results(nB_grid, mean_curve, std_curve, q16, q84,
                 nB_known, cs2_known, samples_phys=None,
                 n_show=20, title="EOS Reconstruction",
                 save_path=None, show_causal_band=True,
                 weights=None, cs2_err=None):
    """
    Plot the mean, 68% CI band, +/- 1 sigma band, optional spaghetti
    samples, and the known data points.  Optionally weights (N,) rescales
    the spaghetti alpha so heavy-weight curves stand out.

    Returns (fig, ax).
    """
    nB_np     = _to_np(nB_grid)
    mean_np   = _to_np(mean_curve)
    std_np    = _to_np(std_curve)
    q16_np    = _to_np(q16)
    q84_np    = _to_np(q84)
    nB_kn_np  = _to_np(nB_known)
    cs2_kn_np = _to_np(cs2_known)
    cs2_err_np = _to_np(cs2_err) if cs2_err is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if show_causal_band:
        _draw_causal_region(
            ax, band_alpha=0.06, edge_lw=0.8,
            band_label=r"Causal region $0\leq c_s^2\leq 1$")
    else:
        _draw_causal_region(ax, draw_band=False)

    if samples_phys is not None:
        samples_np = _to_np(samples_phys)
        n_plot = min(n_show, len(samples_np))
        if weights is not None:
            w_np = _to_np(weights)
            w_rel = w_np[:n_plot] / w_np[:n_plot].max()
            alphas = 0.05 + 0.35 * w_rel
        else:
            alphas = [0.08] * n_plot
        for i in range(n_plot):
            ax.plot(nB_np, samples_np[i], color="steelblue",
                    alpha=float(alphas[i]), linewidth=0.5)

    ax.fill_between(nB_np, q16_np, q84_np,
                    alpha=0.3, color="steelblue",
                    label="68% CI (16th-84th)")

    ax.fill_between(nB_np, mean_np - std_np, mean_np + std_np,
                    alpha=0.15, color="royalblue",
                    label=r"Mean $\pm 1\sigma$")

    ax.plot(nB_np, mean_np, color="darkblue", linewidth=2, label="Mean")

    if cs2_err_np is not None:
        ax.errorbar(nB_kn_np, cs2_kn_np, yerr=cs2_err_np,
                    fmt="o", color="red", markersize=8,
                    markeredgecolor="darkred", markeredgewidth=1.0,
                    ecolor="darkred", elinewidth=1.4, capsize=4,
                    capthick=1.4, zorder=5, label="Known data")
    else:
        ax.scatter(nB_kn_np, cs2_kn_np, color="red", s=60, zorder=5,
                   edgecolors="darkred", linewidths=1.0, label="Known data")

    ax.set_xlabel(r"$n_B / n_0$", fontsize=13)
    ax.set_ylabel(r"$c_s^2$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")

    return fig, ax


# ================================================================
# Plot 2: nested posterior bands
# ================================================================
def plot_nested_posteriors(output, q_low=0.16, q_high=0.84,
                           save_path=None,
                           title="Nested posteriors: progressive constraints"):
    """
    Overlay nested c_s^2 bands showing how each successive layer of
    conditioning narrows the EOS reconstruction:
      chi-EFT (outline)  ->  + astro (warm fill)  ->  + pQCD (cool fill).

    The band stack is recomputed from per-term log-L arrays cached in
    output['reweighting'].

    Returns (fig, ax).
    """
    bands = _nested_posterior_bands(output, q_low=q_low, q_high=q_high)

    nB        = _to_np(output["nB_grid"])
    nB_known  = _to_np(output["nB_known"])
    cs2_known = _to_np(output["cs2_known"])
    cs2_err   = _anchor_err_from_output(output)

    style = {
        "chi_eft": {
            "fc":     "none",
            "alpha":  0.0,
            "ec":     "#555555",
            "lw":     1.2,
            "ls":     ":",
            "med_ls": ":",
            "med_lw": 1.0,
            "z":      2},
        "chi_eft_astro": {
            "fc":     "#fdae6b",
            "alpha":  0.12,
            "ec":     "#d94801",
            "lw":     1.0,
            "ls":     "--",
            "med_ls": "--",
            "med_lw": 1.0,
            "ec_alpha": 0.55,
            "z":      3},
        "chi_eft_pqcd": {
            "fc":     "none",
            "alpha":  0.0,
            "ec":     "#31a354",
            "lw":     1.4,
            "ls":     "-.",
            "med_ls": "-.",
            "med_lw": 1.4,
            "ec_alpha": 0.9,
            "z":      3},
        "chi_eft_astro_pqcd": {
            "fc":     "#3182bd",
            "alpha":  0.50,
            "ec":     "#08519c",
            "lw":     1.8,
            "ls":     "-",
            "med_ls": "-",
            "med_lw": 2.2,
            "z":      4}}

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Faint causal background + conformal-limit reference.
    _draw_causal_region(ax, band_alpha=0.05, edge_lw=0.7)

    order = ["chi_eft", "chi_eft_pqcd", "chi_eft_astro", "chi_eft_astro_pqcd"]
    pct = int(round((q_high - q_low) * 100))

    # The median curve is drawn only for the full posterior; the other
    # constraint layers are represented by their CI band / edges alone.
    full_key = "chi_eft_astro_pqcd" if "chi_eft_astro_pqcd" in bands else "chi_eft_astro"

    for key in order:
        if key not in bands:
            continue
        b = bands[key]
        s = style[key]
        label = f"{b['label']}  [{pct}% CI, ESS={b['ess']:.0f}]"

        if s["fc"] != "none":
            ax.fill_between(nB, b["q_low"], b["q_high"],
                            facecolor=s["fc"], alpha=s["alpha"],
                            edgecolor="none", zorder=s["z"])

        ec_alpha = s.get("ec_alpha", 1.0)
        ax.plot(nB, b["q_low"],  color=s["ec"], linewidth=s["lw"],
                linestyle=s["ls"], zorder=s["z"] + 0.4, alpha=ec_alpha)
        ax.plot(nB, b["q_high"], color=s["ec"], linewidth=s["lw"],
                linestyle=s["ls"], zorder=s["z"] + 0.4, alpha=ec_alpha,
                label=label)

        if key == full_key:
            ax.plot(nB, b["median"], color=s["ec"],
                    linewidth=s["med_lw"], linestyle=s["med_ls"],
                    zorder=s["z"] + 0.6, alpha=ec_alpha)

    if cs2_err is not None:
        ax.errorbar(nB_known, cs2_known, yerr=cs2_err,
                    fmt="o", color="red", markersize=8.5,
                    markeredgecolor="darkred", markeredgewidth=1.0,
                    ecolor="darkred", elinewidth=1.4, capsize=4,
                    capthick=1.4, zorder=20,
                    label=r"$\chi$EFT anchors")
    else:
        ax.scatter(nB_known, cs2_known, color="red", s=70, zorder=20,
                   edgecolor="darkred", linewidth=1.0,
                   label=r"$\chi$EFT anchors")

    ax.set_xlabel(r"$n_B / n_0$", fontsize=13)
    ax.set_ylabel(r"$c_s^2$",     fontsize=13)
    ax.set_title(title,           fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best", framealpha=0.95)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Nested-posteriors plot saved to {save_path}")
    return fig, ax


# ================================================================
# Plot 3: prior vs posterior bands
# ================================================================
def plot_prior_vs_posterior(output, save_path=None,
                            title="Prior vs posterior bands"):
    """
    Single-panel comparison of the unweighted prior band (chi-EFT
    inpainting only) and the full posterior band, both shown as
    mean +/- 1 sigma.

    Where the gray prior band visibly extends past the blue posterior
    band, the data narrowed the constraint there.  Where the two
    overlap, the data did nothing.

    Returns (fig, ax).
    """
    nB         = _to_np(output["nB_grid"])
    prior_mean = _to_np(output["mean_curve"])
    prior_std  = _to_np(output["std_curve"])
    post_mean  = _to_np(output["weighted_mean"])
    post_std   = _to_np(output["weighted_std"])
    nB_known   = _to_np(output["nB_known"])
    cs2_known  = _to_np(output["cs2_known"])
    cs2_err    = _anchor_err_from_output(output)
    full_ess   = float(output["reweighting"]["ESS"])

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_causal_region(ax, band_alpha=0.05, edges=False)

    ax.fill_between(nB, prior_mean - prior_std, prior_mean + prior_std,
                    color="gray", alpha=0.30,
                    label=r"Prior mean $\pm 1\sigma$")
    ax.plot(nB, prior_mean, color="dimgray", linewidth=1.2, linestyle="--")

    ax.fill_between(nB, post_mean - post_std, post_mean + post_std,
                    color="steelblue", alpha=0.45,
                    label=r"Posterior mean $\pm 1\sigma$ "
                          f"(ESS={full_ess:.0f})")
    ax.plot(nB, post_mean, color="darkblue", linewidth=2.0)

    if cs2_err is not None:
        ax.errorbar(nB_known, cs2_known, yerr=cs2_err,
                    fmt="o", color="red", markersize=8,
                    markeredgecolor="darkred", markeredgewidth=1.0,
                    ecolor="darkred", elinewidth=1.4, capsize=4,
                    capthick=1.4, zorder=5, label="Anchors")
    else:
        ax.scatter(nB_known, cs2_known, color="red", s=60, zorder=5,
                   edgecolor="darkred", linewidth=1.0, label="Anchors")

    ax.set_xlabel(r"$n_B / n_0$", fontsize=13)
    ax.set_ylabel(r"$c_s^2$",     fontsize=13)
    ax.set_title(title,           fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Prior-vs-posterior plot saved to {save_path}")
    return fig, ax