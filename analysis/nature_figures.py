"""
Nature-style combined figures for the EOS-diffusion paper.

Assembles the panels that were previously saved as separate PNGs (and
arranged with LaTeX subfigure) into single multi-panel images with
bold, unbracketed panel letters (a, b, c, ...) drawn inside each
panel, matching Nature's figure conventions.

Three public functions, one per paper figure:

  figure_diagnostics_2x2   -> Fig. "cs2_diagnostics"
      a: c_s^2(n_B)   b: gamma(n_B)   c: Delta(mu_B) + Marczenko
      d: d_c(n_B)     (panel order follows the paper caption)

  figure_mr_pfree          -> Fig. "comparison" (M-R | P/P_free)
      a: mass-radius posterior with NICER pulsars and Annala overlays
      b: P/P_free versus mu_B

  figure_fiducials_forest  -> Fig. "fiducials_forest" (2 x 3 grid, a-f)

All drawing of individual panels is delegated to the (patched)
analysis.plots functions, so the per-panel style is pixel-identical to
the previous standalone figures.  This module only arranges panels,
adds the letters, and saves ONE file per figure.

Requires the patched analysis/plots.py in which
_save_single_diagnostic_panel and plot_M_R_band accept ax=None.

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:                                # module dropped inside the package
    from .plots import (_save_single_diagnostic_panel, plot_M_R_band,
                        add_shared_diagnostics_legend, merge_pulsar_legend)
    from .diagnostics import _to_np
except ImportError:                 # module next to the notebook
    from analysis.plots import (_save_single_diagnostic_panel, plot_M_R_band,
                                add_shared_diagnostics_legend,
                                merge_pulsar_legend)
    from analysis.diagnostics import _to_np


# ================================================================
# Panel letters -- Nature style: bold lowercase, no brackets
# ================================================================
def panel_label(ax, letter, *, x=0.03, y=0.97, fontsize=15, ha="left"):
    """Bold, unbracketed panel letter inside a panel corner (Nature style)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            ha=ha, va="top",
            fontsize=fontsize, fontweight="bold", zorder=50)


# ================================================================
# Fig. cs2_diagnostics : 2 x 2  (a: cs2, b: gamma, c: Delta(mu), d: d_c)
# ================================================================
# Panel specs copied verbatim from plot_diagnostics_4panel so the
# single-panel renderer draws exactly what it drew before.
_SPEC_CS2 = ("$c_s^2$",
             dict(ylim=(-0.02, 1.02), causal=True, conformal_at=1.0 / 3.0,
                  show_anchors=True, diag_key="cs2", key="cs2"))
_SPEC_GAMMA = ("$\\gamma = \\mathrm{d}\\ln P / \\mathrm{d}\\ln\\varepsilon$",
               dict(ylim=(0, None), hline_at=1.0,
                    show_anchors=False, diag_key="gamma", key="gamma"))
_SPEC_DC = ("$d_c = \\sqrt{\\Delta^2 + (\\Delta^\\prime)^2}$",
            dict(ylim=(0, None), hline_at=None,
                 show_anchors=False, diag_key="d_c", key="d_c"))


def _draw_delta_mu_panel(ax, mu_grid_MeV,
                         bands_this, bands_gp=None, bands_c4=None,
                         mu_TOV_GeV=None, marczenko=(-0.01, 0.03),
                         gp_label="Annala 2023 GP",
                         c4_label="Annala 2023 C4",
                         gp_color="darkorange", c4_color="purple",
                         label_fontsize=11):
    """Delta(mu_B) panel, replicating notebook section 15d step 5."""
    mu_GeV = np.asarray(mu_grid_MeV, dtype=np.float64) / 1000.0

    ax.fill_between(mu_GeV, bands_this[0], bands_this[2],
                    color="steelblue", alpha=0.30,
                    label="This work (68% CI)")
    ax.plot(mu_GeV, bands_this[1], color="steelblue", lw=2,
            label="This work median")

    if bands_gp is not None:
        ax.fill_between(mu_GeV, bands_gp[0], bands_gp[2],
                        color=gp_color, alpha=0.18,
                        label=f"{gp_label} (68% CI)")
        ax.plot(mu_GeV, bands_gp[1], color=gp_color, lw=1.5, ls="--",
                label=f"{gp_label} median")

    if bands_c4 is not None:
        ax.fill_between(mu_GeV, bands_c4[0], bands_c4[2],
                        color=c4_color, alpha=0.18,
                        label=f"{c4_label} (68% CI)")
        ax.plot(mu_GeV, bands_c4[1], color=c4_color, lw=1.5, ls="--",
                label=f"{c4_label} median")

    ax.axhline(0.0, color="0.3", ls=":", lw=1.0,
               label=r"Conformal limit ($\Delta = 0$)")

    if mu_TOV_GeV is not None:
        ax.axvline(mu_TOV_GeV, color="0.4", ls="-.", lw=1.0,
                   label=(r"$\mu_\mathrm{TOV}$ (this work, median) = "
                          f"{mu_TOV_GeV:.2f} GeV"))
        if marczenko is not None:
            ax.errorbar(mu_TOV_GeV, marczenko[0], yerr=marczenko[1],
                        fmt="D", color="tab:green", ms=7,
                        mec="black", mew=0.6, capsize=4,
                        label=(r"Marczenko+23 PRC: "
                               r"$\Delta_\mathrm{TOV} = -0.01 \pm 0.03$"))

    ax.set_xlabel(r"$\mu_B$ [GeV]", fontsize=label_fontsize)
    ax.set_ylabel(r"$\Delta = 1/3 - P/\varepsilon$", fontsize=label_fontsize)
    ax.set_xlim(0.9, 2.7)
    ax.set_ylim(-0.15, 0.40)
    ax.grid(True, alpha=0.3)
    return ax


def figure_diagnostics_2x2(nB_grid, cs2_arr, gamma_arr, dc_arr, weights, *,
                           nB_known=None, cs2_known=None, cs2_err=None,
                           external_bands=None,
                           mu_grid_MeV=None,
                           delta_bands_this=None,
                           delta_bands_gp=None,
                           delta_bands_c4=None,
                           mu_TOV_GeV=None,
                           marczenko=(-0.01, 0.03),
                           q_low=0.16, q_high=0.84,
                           save_path=None,
                           figsize=(13.0, 8.5), dpi=300,
                           label_fontsize=11, letter_fontsize=15):
    """
    Combined 2x2 diagnostics figure (paper Fig. cs2_diagnostics):

        a  c_s^2(n_B)  with chi-EFT anchors        b  gamma(n_B)
        c  Delta(mu_B) with Marczenko+23 diamond   d  d_c(n_B)

    Panels a, b, d are drawn by the same _save_single_diagnostic_panel
    used for the previous standalone PNGs; panel c replicates section
    15d of the notebook.  The single shared legend sits on panel c (as
    in the previous eos_EFT_Delta_vs_mu.png).  Bold unbracketed letters
    are placed inside each panel; the LaTeX side needs no subfigures.

    delta_bands_* are the (q16, q50, q84) tuples computed in notebook
    section 15d (bands_D_this, bands_D_gp, bands_D_c4) on mu_grid_MeV
    (= mu_grid_common, in MeV).
    """
    nB = _to_np(nB_grid)
    w  = _to_np(weights).astype(np.float64)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # a -- c_s^2 (with anchors)
    ylab, opts = _SPEC_CS2
    _save_single_diagnostic_panel(
        nB, _to_np(cs2_arr).astype(np.float64), w, ylab, opts,
        nB_known=nB_known, cs2_known=cs2_known, cs2_err=cs2_err,
        external_bands=external_bands, q_low=q_low, q_high=q_high,
        save_path=None, label_fontsize=label_fontsize,
        draw_legend=False, ax=axes[0, 0])

    # b -- gamma
    ylab, opts = _SPEC_GAMMA
    _save_single_diagnostic_panel(
        nB, np.asarray(gamma_arr, dtype=np.float64), w, ylab, opts,
        external_bands=external_bands, q_low=q_low, q_high=q_high,
        save_path=None, label_fontsize=label_fontsize,
        draw_legend=False, ax=axes[0, 1])

    # c -- Delta(mu_B) + Marczenko, carrying the single shared legend
    _draw_delta_mu_panel(axes[1, 0], mu_grid_MeV,
                         delta_bands_this, delta_bands_gp, delta_bands_c4,
                         mu_TOV_GeV=mu_TOV_GeV, marczenko=marczenko,
                         label_fontsize=label_fontsize)
    if external_bands is not None:
        add_shared_diagnostics_legend(
            axes[1, 0],
            external_bands=external_bands,
            chi_eft_color="red",
            marczenko_color="tab:green",
            loc="upper right", ncol=1, fontsize=8)

    # d -- d_c
    ylab, opts = _SPEC_DC
    _save_single_diagnostic_panel(
        nB, np.asarray(dc_arr, dtype=np.float64), w, ylab, opts,
        external_bands=external_bands, q_low=q_low, q_high=q_high,
        save_path=None, label_fontsize=label_fontsize,
        draw_legend=False, ax=axes[1, 1])

    for ax, letter in zip(axes.flat, "abcd"):
        panel_label(ax, letter, fontsize=letter_fontsize)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Nature 2x2 diagnostics figure saved to {save_path}")
    return fig, axes


# ================================================================
# Fig. comparison : 1 x 2  (a: M-R, b: P/P_free)
# ================================================================
def _draw_pfree_panel(ax, mu_grid_MeV,
                      bands_this, bands_gp=None, bands_c4=None,
                      annala_quoted=(0.37, 0.43), pqcd_anchor_GeV=2.6,
                      gp_label="Annala 2023 GP", c4_label="Annala 2023 C4",
                      gp_color="darkorange", c4_color="purple",
                      label_fontsize=11, legend_fontsize=8):
    """P/P_free(mu_B) panel, replicating notebook section 15c step 4."""
    mu_GeV = np.asarray(mu_grid_MeV, dtype=np.float64) / 1000.0

    ax.fill_between(mu_GeV, bands_this[0], bands_this[2],
                    color="steelblue", alpha=0.30,
                    label="This work (68% CI)")
    ax.plot(mu_GeV, bands_this[1], color="steelblue", lw=2,
            label="This work median")

    if bands_gp is not None:
        ax.fill_between(mu_GeV, bands_gp[0], bands_gp[2],
                        color=gp_color, alpha=0.18,
                        label=f"{gp_label} (68% CI)")
        ax.plot(mu_GeV, bands_gp[1], color=gp_color, lw=1.5, ls="--",
                label=f"{gp_label} median")

    if bands_c4 is not None:
        ax.fill_between(mu_GeV, bands_c4[0], bands_c4[2],
                        color=c4_color, alpha=0.18,
                        label=f"{c4_label} (68% CI)")
        ax.plot(mu_GeV, bands_c4[1], color=c4_color, lw=1.5, ls="--",
                label=f"{c4_label} median")

    if annala_quoted is not None:
        ax.axhspan(annala_quoted[0], annala_quoted[1],
                   color="gray", alpha=0.12,
                   label=(r"Annala+23 quoted: "
                          r"$P/P^\mathrm{free}(\mu_\mathrm{TOV}) "
                          r"= 0.40 \pm 0.03$"))
    if pqcd_anchor_GeV is not None:
        ax.axvline(pqcd_anchor_GeV, color="0.4", ls=":", lw=1.2,
                   label=(r"pQCD anchor: $\mu_B = "
                          f"{pqcd_anchor_GeV:.1f}$ GeV"))

    ax.set_xlabel(r"$\mu_B$ [GeV]", fontsize=label_fontsize)
    ax.set_ylabel(r"$P / P^\mathrm{free}$", fontsize=label_fontsize)
    ax.set_xlim(0.9, 2.7)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fontsize, loc="upper left", framealpha=0.95)
    return ax


def figure_mr_pfree(M_arr, R_arr, weights, *,
                    observed_pulsars=None,
                    mr_external_bands=None,
                    c4_mass_grid=None, c4_R_bands=None,
                    mtov_bound=1.908,
                    mtov_bound_label=(r"$M_\mathrm{TOV} \geq "
                                      r"1.908\,M_\odot$ (J1614-2230)"),
                    mu_grid_MeV=None,
                    pfree_bands_this=None,
                    pfree_bands_gp=None,
                    pfree_bands_c4=None,
                    annala_quoted=(0.37, 0.43), pqcd_anchor_GeV=2.6,
                    q_low=0.16, q_high=0.84,
                    min_weight_frac=0.30, M_max_quantile=0.90,
                    save_path=None,
                    figsize=(14.0, 5.5), dpi=300,
                    letter_fontsize=15):
    """
    Combined 1x2 figure (paper Fig. comparison):

        a  M-R posterior + NICER pulsars + Annala GP contour + C4 band
        b  P/P_free versus mu_B with the Annala+23 headline band

    Panel a is drawn by the patched plot_M_R_band (identical style to
    eos_EFT_M_R_publication.png), then decorated exactly as notebook
    section 15a: C4 overlay, M_TOV >= 1.908 line, and the merged
    numbered-pulsar legend.  Panel b replicates section 15c.

    c4_R_bands       : (q16, q50, q84) of the C4 R(M) on c4_mass_grid.
    pfree_bands_*    : (q16, q50, q84) tuples on mu_grid_MeV.
    """
    fig, (ax_mr, ax_mu) = plt.subplots(1, 2, figsize=figsize)

    # ---- a: M-R ----------------------------------------------------
    plot_M_R_band(M_arr, R_arr, weights,
                  observed_pulsars=observed_pulsars,
                  external_bands=mr_external_bands,
                  save_path=None,
                  q_low=q_low, q_high=q_high,
                  min_weight_frac=min_weight_frac,
                  M_max_quantile=M_max_quantile,
                  ax=ax_mr)

    if c4_mass_grid is not None and c4_R_bands is not None:
        ax_mr.fill_betweenx(c4_mass_grid, c4_R_bands[0], c4_R_bands[2],
                            color="purple", alpha=0.18,
                            label="Annala 2023 (C4) 68% CI", zorder=2)
        ax_mr.plot(c4_R_bands[1], c4_mass_grid,
                   color="purple", lw=1.5, ls="--",
                   label="Annala 2023 (C4) median", zorder=3)

    if mtov_bound is not None:
        ax_mr.axhline(mtov_bound, color="0.5", ls=":", lw=1, alpha=0.6,
                      label=mtov_bound_label)

    # Rebuild the legend so C4 + M_TOV join the numbered NICER entries
    handles, labels = ax_mr.get_legend_handles_labels()
    handles, labels, handler_map = merge_pulsar_legend(
        ax_mr, handles, labels)
    leg = ax_mr.get_legend()
    if leg is not None:
        leg.remove()
    ax_mr.legend(handles, labels,
                 handler_map=handler_map,
                 loc="upper left", fontsize=7, framealpha=0.95,
                 labelspacing=0.6, handlelength=1.6,
                 handletextpad=0.5, borderpad=0.3, borderaxespad=0.3)

    # ---- b: P/P_free ------------------------------------------------
    _draw_pfree_panel(ax_mu, mu_grid_MeV,
                      pfree_bands_this, pfree_bands_gp, pfree_bands_c4,
                      annala_quoted=annala_quoted,
                      pqcd_anchor_GeV=pqcd_anchor_GeV)

    panel_label(ax_mr, "a", fontsize=letter_fontsize, x=0.97, ha="right")
    panel_label(ax_mu, "b", fontsize=letter_fontsize, x=0.97, ha="right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Nature M-R + P/P_free figure saved to {save_path}")
    return fig, (ax_mr, ax_mu)


# ================================================================
# Fig. fiducials_forest : 2 x 3 grid, letters a-f
# ================================================================
def figure_fiducials_forest(quantities, ensemble_rows, lit_numeric,
                            weighted_quantiles_fn, *,
                            xlabels=None, skip_ensemble=None,
                            panels_per_row=3,
                            save_path=None, dpi=300,
                            letter_fontsize=15,
                            top_headroom=0.9):
    """
    Consolidated forest plot with bold unbracketed panel letters.
    Same layout and per-row styling as the notebook's
    plot_fiducials_forest; per-panel standalone saving is removed.

    ensemble_rows : list of (label, per_sample_dict, weights, color),
        e.g. [("this work", per_sample_this, weights, "C0"), ...]
    lit_numeric   : the LIT_NUMERIC dict of published values.
    weighted_quantiles_fn : the notebook's _weighted_quantiles helper,
        called as fn(values, weights, qs=(0.16, 0.5, 0.84)).
    """
    skip_ensemble = skip_ensemble or {}

    def _stats(ps, w, key):
        if key not in ps:
            return None
        q_lo, med, q_hi = weighted_quantiles_fn(
            ps[key], w, qs=(0.16, 0.5, 0.84))
        if not np.isfinite(med):
            return None
        return (med, q_lo, q_hi)

    nQ   = len(quantities)
    ncol = panels_per_row
    nrow = (nQ + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(4.5 * ncol, 2.0 + 0.55 * nrow * 6))
    axes = np.atleast_2d(axes).flatten()

    for ax_i, q in enumerate(quantities):
        ax = axes[ax_i]
        labels, positions = [], []
        y = 0
        for label, ps, w, color in ensemble_rows:
            if q in skip_ensemble.get(label, set()):
                continue
            stats = _stats(ps, w, q)
            if stats is None:
                continue
            med, q16, q84 = stats
            ax.errorbar(med, y, xerr=[[med - q16], [q84 - med]],
                        fmt='o', color=color, capsize=4, lw=2,
                        markersize=7)
            labels.append(label)
            positions.append(y)
            y += 1
        for label, stats in lit_numeric.get(q, {}).items():
            med, q16, q84 = stats
            ax.errorbar(med, y, xerr=[[med - q16], [q84 - med]],
                        fmt='s', color='gray', capsize=4, lw=1.5,
                        markerfacecolor='white', markersize=7)
            labels.append(label)
            positions.append(y)
            y += 1

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        unit = (xlabels or {}).get(q, "")
        ax.set_xlabel(f"{q}" + (f" [{unit}]" if unit else ""), fontsize=9)
        # Open a gap of `top_headroom` rows ABOVE the top point (y=0) so the
        # bold panel letter has clear space and never overlaps the topmost
        # ("this work") error bar.  Axis is inverted, so the smaller number
        # is the visual top.
        ax.set_ylim(y - 0.5, -0.5 - top_headroom)
        # Letter sits inside that gap, left-aligned, vertically centred in
        # the headroom band rather than at the very top of the axes.
        panel_label(ax, chr(ord("a") + ax_i), fontsize=letter_fontsize,
                    x=0.03, y=0.985)

    for i in range(nQ, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Nature forest figure saved to {save_path}")
    return fig, axes
