"""
Visualization layer for analysis.diagnostics.

All functions are pure visualization on the arrays produced by
analysis.diagnostics; no sampling, no TOV, no pQCD recomputation.
The aesthetic deliberately matches eos_sampling.plotting:
  * steelblue fill / darkblue mean for the posterior band,
  * gray faint causal background where applicable,
  * conformal limit drawn as a thin dashed line,
  * red markers with darkred edges for chi-EFT anchors,
  * all functions return (fig, axes) so the notebook can tweak.

"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from eos_sampling.reweighting import weighted_quantile

from .diagnostics import _to_np, _mr_band


def _norm_w(w):
    """Normalise importance weights to sum to 1; fall back to uniform."""
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / w.size)


def _wq_curve(arr2d, w, q):
    """Pointwise weighted q-quantile down axis 0 -> (L,) curve."""
    L = arr2d.shape[1]
    return np.array([weighted_quantile(arr2d[:, j], w, q) for j in range(L)])


def _draw_causal_band(ax, *, alpha=0.05, edges=True, edge_lw=0.7):
    """Faint gray causal region 0 <= c_s^2 <= 1, with optional dotted edges.
    Reference lines (conformal limit etc.) are drawn separately by callers."""
    ax.axhspan(0.0, 1.0, color="gray", alpha=alpha)
    if edges:
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=edge_lw)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=edge_lw)


def _weighted_mean_std_q(arr, w, q_low=0.16, q_high=0.84):
    """
    Compute weighted mean, std, and (q_low, q_high) quantiles pointwise
    along axis 0.

    arr : (N, L) numpy array
    w   : (N,)  importance weights (need not be normalised)

    Returns mean, std, q_lo, q_hi  (each shape (L,))
    """
    arr = np.asarray(arr, dtype=np.float64)
    w   = np.asarray(w,   dtype=np.float64)
    wn  = _norm_w(w)

    mean = (wn[:, None] * arr).sum(axis=0)
    var  = (wn[:, None] * (arr - mean[None, :]) ** 2).sum(axis=0)
    std  = np.sqrt(var)
    q_lo = _wq_curve(arr, w, q_low)
    q_hi = _wq_curve(arr, w, q_high)

    return mean, std, q_lo, q_hi


def plot_diagnostics_4panel(nB_grid, cs2_arr, gamma_arr, Delta_arr, dc_arr,
                            weights, nB_known=None, cs2_known=None,
                            cs2_err=None, save_path=None,
                            panel_save_paths=None,
                            title="EOS conformality diagnostics",
                            q_low=0.16, q_high=0.84,
                            external_bands=None,
                            panel_figsize=(8.5, 5.5),
                            panel_label_fontsize=11,
                            panel_dpi=150,
                            panel_chi_eft_color="red",
                            panel_draw_legend=False):
    """
    Four-panel figure showing the weighted posterior of:
        (a) c_s^2,                   with chi-EFT anchors marked,
        (b) gamma = d ln P / d ln eps,
        (c) Delta = 1/3 - P/eps,
        (d) d_c   = sqrt(Delta^2 + (Delta')^2).

    Each panel shows the 68% credible interval (steelblue fill), the
    +/-1 sigma band (lighter steelblue), and the weighted mean (darkblue).
    The conformal limit is drawn as a thin dashed gray line on (a) and (c).

    Optional `external_bands` is a list of dicts produced by
    analysis.comparisons.compute_external_band; each is overlaid as a
    contour-only band (median solid line + dashed q_low/q_high lines)
    in the band's color, with no fill, so our posterior remains
    the primary visual element.

    Optional `panel_save_paths`: dict mapping panel keys
    {"cs2", "gamma", "Delta", "d_c"} to file paths.  When given, each
    panel is also rendered as its own standalone figure (no panel label,
    no shared-x cropping, individual legend on the c_s^2 panel only) and
    saved to the corresponding path.  Useful when the LaTeX side wants
    four separate \\includegraphics arranged with subfigure / subfloat.
    The composite figure is still produced and (if save_path is given)
    saved as before; the per-panel files are an extra output.

    Returns (fig, axes).
    """
    nB  = _to_np(nB_grid)
    cs2 = _to_np(cs2_arr).astype(np.float64)
    g   = np.asarray(gamma_arr, dtype=np.float64)
    D   = np.asarray(Delta_arr, dtype=np.float64)
    dc  = np.asarray(dc_arr,    dtype=np.float64)
    w   = _to_np(weights).astype(np.float64)

    panels = [
        ("$c_s^2$",                                 cs2,
         dict(ylim=(-0.02, 1.02),
              causal=True, conformal_at=1.0/3.0,
              show_anchors=True, diag_key="cs2",
              key="cs2")),
        ("$\\gamma = \\mathrm{d}\\ln P / \\mathrm{d}\\ln\\varepsilon$",
                                                          g,
         dict(ylim=(0, None),
              hline_at=1.0,
              show_anchors=False, diag_key="gamma",
              key="gamma")),
        ("$\\Delta = 1/3 - P/\\varepsilon$",         D,
         dict(ylim=None,
              hline_at=0.0,
              show_anchors=False, diag_key="Delta",
              key="Delta")),
        ("$d_c = \\sqrt{\\Delta^2 + (\\Delta^\\prime)^2}$",
                                                          dc,
         dict(ylim=(0, None),
              hline_at=None,
              show_anchors=False, diag_key="d_c",
              key="d_c")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    axes_flat = axes.flat

    for ax, (ylabel, arr, opts) in zip(axes_flat, panels):
        _, _, q_lo, q_hi = _weighted_mean_std_q(
            arr, w, q_low, q_high)
        med = _wq_curve(np.asarray(arr, dtype=np.float64), w, 0.50)

        if opts.get("causal", False):
            _draw_causal_band(ax)

        if opts.get("conformal_at") is not None:
            ax.axhline(opts["conformal_at"], color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5, zorder=1)
        elif opts.get("hline_at") is not None:
            ax.axhline(opts["hline_at"], color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5, zorder=1)

        ax.fill_between(nB, q_lo, q_hi, color="steelblue", alpha=0.30,
                        edgecolor="none", label="This work (68% CI)",
                        zorder=2)
        ax.plot(nB, med, color="steelblue", linewidth=2.0,
                label="This work median", zorder=3)

        if opts.get("show_anchors", False) \
                and nB_known is not None and cs2_known is not None:
            nB_kn  = _to_np(nB_known)
            cs2_kn = _to_np(cs2_known)
            if cs2_err is not None:
                err = _to_np(cs2_err)
                ax.errorbar(nB_kn, cs2_kn, yerr=err,
                            fmt="o", color="red", markersize=7,
                            markeredgecolor="darkred", markeredgewidth=1.0,
                            ecolor="darkred", elinewidth=1.4, capsize=4,
                            capthick=1.4, zorder=20,
                            label=r"$\chi$EFT anchors")
            else:
                ax.scatter(nB_kn, cs2_kn, color="red", s=60, zorder=20,
                           edgecolor="darkred", linewidth=1.0,
                           label=r"$\chi$EFT anchors")

        if external_bands:
            diag_key = opts.get("diag_key")
            for k, ext_b in enumerate(external_bands):
                if diag_key is None or diag_key not in ext_b:
                    continue
                ext_grid = _to_np(ext_b.get("target_grid", nB))
                q_lo_e, q_md_e, q_hi_e = ext_b[diag_key]
                color = ext_b.get("color", f"C{k+2}")
                lbl   = (ext_b.get("label", f"external {k+1}")
                         if opts.get("show_anchors", False) else None)
                ax.plot(ext_grid, q_lo_e, color=color, linestyle="--",
                        linewidth=1.5, alpha=0.9, zorder=4)
                ax.plot(ext_grid, q_hi_e, color=color, linestyle="--",
                        linewidth=1.5, alpha=0.9, zorder=4)
                ax.plot(ext_grid, q_md_e, color=color, linestyle="-",
                        linewidth=1.8, alpha=0.95, zorder=4, label=lbl)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        if opts.get("ylim") is not None:
            yl = opts["ylim"]
            ax.set_ylim(bottom=yl[0], top=yl[1])

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$n_B / n_0$", fontsize=12)
    for ax_i, ax in enumerate(axes_flat_for_labels := list(axes.flat)):
        ax.text(0.03, 0.97, f"({chr(ord('a') + ax_i)})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    _handles, _labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(_handles, _labels, fontsize=9, loc="upper right",
                      framealpha=0.95)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Diagnostics 4-panel saved to {save_path}")
    if panel_save_paths:
        for (ylabel, arr, opts) in panels:
            key = opts.get("key")
            out_path = panel_save_paths.get(key) if key else None
            if not out_path:
                continue
            _save_single_diagnostic_panel(
                nB, arr, w, ylabel, opts,
                nB_known=nB_known, cs2_known=cs2_known, cs2_err=cs2_err,
                external_bands=external_bands,
                q_low=q_low, q_high=q_high,
                save_path=out_path,
                figsize=panel_figsize, label_fontsize=panel_label_fontsize,
                dpi=panel_dpi, chi_eft_color=panel_chi_eft_color,
                draw_legend=panel_draw_legend,
            )
    return fig, axes


def _save_single_diagnostic_panel(nB, arr, w, ylabel, opts,
                                  nB_known=None, cs2_known=None, cs2_err=None,
                                  external_bands=None,
                                  q_low=0.16, q_high=0.84,
                                  save_path=None,
                                  figsize=(8.5, 5.5), label_fontsize=11,
                                  dpi=150, chi_eft_color="red",
                                  draw_legend=False,
                                  ax=None):
    """Render one diagnostic panel as a standalone figure (no panel
    letter, own x-axis), and save to ``save_path``.  Used by
    plot_diagnostics_4panel when panel_save_paths is given.

    The aesthetic deliberately matches eos_EFT_Delta_vs_mu.png so that
    diagnostics_a_cs2 / diagnostics_b_gamma / diagnostics_d_dc and the
    Delta(mu_B) figure can sit together in a single LaTeX figure:

      * same canvas (figsize, default 8.5 x 5.5 in) and same dpi (150),
      * this-work shown as a filled steelblue 68% CI band + a steelblue
        weighted-MEDIAN line (lw=2) -- not the mean/+-1sigma style,
      * each external band filled in its own colour (darkorange GP,
        purple C4) with a dashed weighted-median line,
      * chi-EFT anchors drawn in `chi_eft_color` (red by default) so the
        Marczenko diamond in the Delta figure -- a different colour -- is
        distinguishable in the single SHARED legend,
      * no per-panel legend (draw_legend=False); the four panels share
        one legend built by make_shared_diagnostics_legend().
    """
    pct  = int(round((q_high - q_low) * 100))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        _own_fig = True
    else:
        fig = ax.figure
        _own_fig = False
    _, _, q_lo, q_hi = _weighted_mean_std_q(arr, w, q_low, q_high)
    arr_np = np.asarray(arr, dtype=np.float64)
    med = _wq_curve(arr_np, w, 0.50)

    if opts.get("causal", False):
        _draw_causal_band(ax)
    if opts.get("conformal_at") is not None:
        ax.axhline(opts["conformal_at"], color="0.3", linestyle=":",
                   linewidth=1.0, zorder=1)
    elif opts.get("hline_at") is not None:
        ax.axhline(opts["hline_at"], color="0.3", linestyle=":",
                   linewidth=1.0, zorder=1)
    ax.fill_between(nB, q_lo, q_hi, color="steelblue", alpha=0.30,
                    edgecolor="none", label="This work (68% CI)",
                    zorder=2)
    ax.plot(nB, med, color="steelblue", linewidth=2.0,
            label="This work median", zorder=3)

    if opts.get("show_anchors", False) \
            and nB_known is not None and cs2_known is not None:
        nB_kn  = _to_np(nB_known)
        cs2_kn = _to_np(cs2_known)
        if cs2_err is not None:
            err = _to_np(cs2_err)
            ax.errorbar(nB_kn, cs2_kn, yerr=err,
                        fmt="o", color=chi_eft_color, markersize=7,
                        markeredgecolor="darkred", markeredgewidth=1.0,
                        ecolor="darkred", elinewidth=1.4, capsize=4,
                        capthick=1.4, zorder=20,
                        label=r"$\chi$EFT anchors")
        else:
            ax.scatter(nB_kn, cs2_kn, color=chi_eft_color, s=60, zorder=20,
                       edgecolor="darkred", linewidth=1.0,
                       label=r"$\chi$EFT anchors")

    if external_bands:
        diag_key = opts.get("diag_key")
        for k, ext_b in enumerate(external_bands):
            if diag_key is None or diag_key not in ext_b:
                continue
            ext_grid = _to_np(ext_b.get("target_grid", nB))
            q_lo_e, q_md_e, q_hi_e = ext_b[diag_key]
            color = ext_b.get("color", f"C{k+2}")
            lbl   = ext_b.get("label", f"external {k+1}")
            ax.fill_between(ext_grid, q_lo_e, q_hi_e, color=color,
                            alpha=0.18, edgecolor="none",
                            label=f"{lbl} (68% CI)", zorder=4)
            ax.plot(ext_grid, q_md_e, color=color, linestyle="--",
                    linewidth=1.5, alpha=0.95, zorder=5,
                    label=f"{lbl} median")

    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel(r"$n_B / n_0$", fontsize=label_fontsize)
    ax.grid(True, alpha=0.3)
    if opts.get("ylim") is not None:
        yl = opts["ylim"]
        ax.set_ylim(bottom=yl[0], top=yl[1])
    if draw_legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)

    if _own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  diagnostic panel saved to {save_path}")
    return ax


def _shared_diagnostics_handles(external_bands=None,
                                include_chi_eft=True,
                                include_conformal=True,
                                include_mu_tov=True,
                                include_marczenko=True,
                                chi_eft_color="red",
                                marczenko_color="tab:green",
                                marczenko_label=(r"Marczenko+23: "
                                                 r"$\Delta_\mathrm{TOV} = "
                                                 r"-0.01 \pm 0.03$")):
    """Build the colour-matched handle list shared by the four diagnostic
    panels (c_s^2, gamma, d_c, and Delta(mu_B)).  Used by both
    make_shared_diagnostics_legend (standalone strip) and
    add_shared_diagnostics_legend (legend placed on an existing axis)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="steelblue", alpha=0.30, label="This work (68% CI)"),
        Line2D([0], [0], color="steelblue", lw=2.0, label="This work median"),
    ]
    if external_bands:
        for k, ext_b in enumerate(external_bands):
            color = ext_b.get("color", f"C{k+2}")
            lbl   = ext_b.get("label", f"external {k+1}")
            handles.append(Patch(facecolor=color, alpha=0.18,
                                 label=f"{lbl} (68% CI)"))
            handles.append(Line2D([0], [0], color=color, lw=1.5, ls="--",
                                  label=f"{lbl} median"))
    if include_chi_eft:
        handles.append(Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor=chi_eft_color,
                              markeredgecolor="darkred", markeredgewidth=1.0,
                              markersize=8, label=r"$\chi$EFT anchors"))
    if include_conformal:
        handles.append(Line2D([0], [0], color="0.3", ls=":", lw=1.0,
                              label=r"Conformal limit"))
    if include_mu_tov:
        handles.append(Line2D([0], [0], color="0.4", ls="-.", lw=1.0,
                              label=r"$\mu_\mathrm{TOV}$ (this work, median)"))
    if include_marczenko:
        handles.append(Line2D([0], [0], marker="D", linestyle="none",
                              markerfacecolor=marczenko_color,
                              markeredgecolor="black", markeredgewidth=0.6,
                              markersize=8, label=marczenko_label))
    return handles


def add_shared_diagnostics_legend(ax, external_bands=None,
                                  include_chi_eft=True,
                                  include_conformal=True,
                                  include_mu_tov=True,
                                  include_marczenko=True,
                                  chi_eft_color="red",
                                  marczenko_color="tab:green",
                                  marczenko_label=(r"Marczenko+23: "
                                                   r"$\Delta_\mathrm{TOV} = "
                                                   r"-0.01 \pm 0.03$"),
                                  loc="upper right", ncol=1, fontsize=8,
                                  framealpha=0.95, **legend_kw):
    """
    Place the single SHARED legend for all four diagnostic panels
    directly on an existing axis -- intended for the Delta(mu_B) panel,
    which has the most empty space (the bands descend to the right).

    The legend is built from explicit colour-matched proxies (NOT the
    axis' own artists), so it always carries every entry that appears
    across the four panels -- including the chi-EFT anchors, which are
    only physically drawn on the c_s^2 panel.  The Marczenko diamond is
    drawn in `marczenko_color` (default tab:green) so it stays
    distinguishable from the red chi-EFT anchors in the same box.

    Parameters
    ----------
    ax : matplotlib Axes to attach the legend to (e.g. ax_D).
    external_bands : the same band dicts passed to plot_diagnostics_4panel.
    loc, ncol, fontsize, framealpha, **legend_kw : forwarded to ax.legend.

    Returns
    -------
    the Legend object.
    """
    handles = _shared_diagnostics_handles(
        external_bands=external_bands,
        include_chi_eft=include_chi_eft,
        include_conformal=include_conformal,
        include_mu_tov=include_mu_tov,
        include_marczenko=include_marczenko,
        chi_eft_color=chi_eft_color,
        marczenko_color=marczenko_color,
        marczenko_label=marczenko_label,
    )
    return ax.legend(handles=handles, loc=loc, ncol=ncol,
                     fontsize=fontsize, framealpha=framealpha, **legend_kw)


def make_shared_diagnostics_legend(save_path,
                                   external_bands=None,
                                   include_chi_eft=True,
                                   include_conformal=True,
                                   include_mu_tov=True,
                                   include_marczenko=True,
                                   chi_eft_color="red",
                                   marczenko_color="tab:green",
                                   marczenko_label=(r"Marczenko+23: "
                                                    r"$\Delta_\mathrm{TOV} = "
                                                    r"-0.01 \pm 0.03$"),
                                   ncol=4, figsize=(12.0, 0.9), fontsize=9,
                                   dpi=150):
    """
    Build the shared four-panel legend as its own little strip image
    (an alternative to add_shared_diagnostics_legend, which puts the
    same legend directly on the Delta panel).  Colour conventions match
    _save_single_diagnostic_panel and the Delta(mu_B) cell.

    Returns save_path (for convenience).
    """
    handles = _shared_diagnostics_handles(
        external_bands=external_bands,
        include_chi_eft=include_chi_eft,
        include_conformal=include_conformal,
        include_mu_tov=include_mu_tov,
        include_marczenko=include_marczenko,
        chi_eft_color=chi_eft_color,
        marczenko_color=marczenko_color,
        marczenko_label=marczenko_label,
    )
    fig = plt.figure(figsize=figsize)
    fig.legend(handles=handles, loc="center", ncol=ncol,
               fontsize=fontsize, framealpha=0.95,
               handlelength=1.8, handletextpad=0.5,
               columnspacing=1.4, labelspacing=0.5)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  shared diagnostics legend saved to {save_path}")
    return save_path


def plot_P_eps_band(P_arr, eps_arr, weights, eps_grid=None,
                    save_path=None,
                    title=r"Pressure-energy density posterior",
                    q_low=0.16, q_high=0.84):
    """
    Weighted posterior P(eps) band.  Each sample's (eps, P) is
    interpolated onto a common log-spaced eps grid before pointwise
    quantiles are computed.

    Returns (fig, ax).
    """
    P   = _to_np(P_arr).astype(np.float64)
    eps = _to_np(eps_arr).astype(np.float64)
    w   = _to_np(weights).astype(np.float64)
    N = P.shape[0]

    if eps_grid is None:
        eps_lo = float(np.percentile(eps[:, 0],  90))
        eps_hi = float(np.percentile(eps[:, -1], 10))
        if eps_hi <= eps_lo:
            eps_lo, eps_hi = float(eps.min()), float(eps.max())
        eps_grid = np.logspace(np.log10(eps_lo), np.log10(eps_hi), 200)

    L = len(eps_grid)
    P_on_grid = np.full((N, L), np.nan)
    for n in range(N):
        e_n, p_n = eps[n], P[n]
        order = np.argsort(e_n)
        e_n, p_n = e_n[order], p_n[order]
        valid = (eps_grid >= e_n.min()) & (eps_grid <= e_n.max())
        if not valid.any():
            continue
        P_on_grid[n, valid] = np.interp(eps_grid[valid], e_n, p_n)

    pct = int(round((q_high - q_low) * 100))
    q_lo = _wq_curve(P_on_grid, w, q_low)
    q_md = _wq_curve(P_on_grid, w, 0.50)
    q_hi = _wq_curve(P_on_grid, w, q_high)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(eps_grid, q_lo, q_hi, color="steelblue", alpha=0.35,
                    label=f"{pct}% CI")
    ax.plot(eps_grid, q_md, color="darkblue", linewidth=2.0, label="Median")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$ [MeV/fm$^3$]", fontsize=12)
    ax.set_ylabel(r"$P$ [MeV/fm$^3$]",          fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best", framealpha=0.95)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  P-eps band saved to {save_path}")
    return fig, ax


def merge_pulsar_legend(ax, handles=None, labels=None, handler_map=None):
    """Merge stashed numbered-pulsar legend entries into a fresh handle list.

    plot_M_R_band attaches the numbered-pulsar proxies and custom legend
    handler_map on the axes as ``ax._pulsar_legend_*`` attributes.  Cells
    that rebuild the legend later -- e.g. after adding a C4 band or an
    M_TOV reference line -- should call this helper to merge those
    numbered entries back into the new legend.

    Usage:
        handles, labels = ax.get_legend_handles_labels()
        handles, labels, handler_map = merge_pulsar_legend(ax, handles, labels)
        ax.legend(handles, labels, handler_map=handler_map, ...)

    If ``handles``/``labels`` are omitted, they are read from the axes.
    Returns ``(handles, labels, handler_map)`` always.
    """
    if handles is None or labels is None:
        h, l = ax.get_legend_handles_labels()
        if handles is None: handles = h
        if labels  is None: labels  = l
    handles = list(handles)
    labels  = list(labels)
    handler_map = dict(handler_map or {})
    proxies = getattr(ax, "_pulsar_legend_proxies", None)
    if proxies:
        handles.extend(proxies)
        labels.extend(getattr(ax, "_pulsar_legend_labels", []))
        handler_map.update(getattr(ax, "_pulsar_legend_handler_map", {}))
    return handles, labels, handler_map


def plot_M_R_band(M_arr, R_arr, weights, M_grid=None,
                  observed_pulsars=None, save_path=None,
                  title=r"Mass-radius posterior",
                  q_low=0.16, q_high=0.84,
                  external_bands=None,
                  min_weight_frac=0.30,
                  M_max_quantile=0.90,
                  ax=None):
    """
    Weighted posterior M-R band built from the cached TOV arrays.
    Each sample's stable branch is interpolated onto a common M grid;
    pointwise weighted quantiles produce the band.

    observed_pulsars : optional list of dicts with keys
        {"name", "M_obs", "sigma_M", "R_obs", "sigma_R"}
        -- the same schema used by the reweighter -- to overlay
        NICER measurements.

    external_bands : optional list of dicts produced by
        analysis.comparisons.compute_external_M_R_band; each is
        overlaid as a contour-only band (median solid line + dashed
        q_low/q_high lines) in the band's color, no fill.

    min_weight_frac : float in [0, 1].  At each M on the grid, the
        weighted fraction of samples whose stable branch reaches M is
        computed; when this fraction falls below min_weight_frac, the
        band is masked at that M (NaN-filled and dropped from the
        figure).  Default 0.05.  Prevents the high-mass tail from being
        contaminated by the handful of samples whose M_TOV reaches
        far above the bulk -- the typical cause of high-mass band
        instability.  Set to 0 to disable the mask.

    M_max_quantile : float in (0, 1].  Upper edge of the M grid is
        chosen as the weighted M_max_quantile-quantile of M_TOV across
        the ensemble (default 0.95).  Override by passing M_grid
        directly.  Set to None to use a fixed default range.

    Returns (fig, ax).
    """
    M = _to_np(M_arr).astype(np.float64)
    R = _to_np(R_arr).astype(np.float64)
    w = _to_np(weights).astype(np.float64)
    N = M.shape[0]

    M_TOV_per = np.array([np.nanmax(M[n]) if np.any(np.isfinite(M[n])) else np.nan
                          for n in range(N)])

    if M_grid is None:
        if M_max_quantile is not None:
            finite = np.isfinite(M_TOV_per)
            M_top = float(weighted_quantile(M_TOV_per[finite],
                                            w[finite], M_max_quantile))
        else:
            M_top = 2.6
        M_grid = np.linspace(0.6, M_top, 200)
    pct = int(round((q_high - q_low) * 100))
    band, weight_frac = _mr_band(M, R, w, M_grid, q_low=q_low, q_high=q_high)
    q_lo, q_md, q_hi = band

    mask_low = weight_frac < float(min_weight_frac)
    if mask_low.any():
        q_lo[mask_low] = np.nan
        q_md[mask_low] = np.nan
        q_hi[mask_low] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        _own_fig = True
    else:
        fig = ax.figure
        _own_fig = False
    ax.fill_betweenx(M_grid, q_lo, q_hi, color="steelblue", alpha=0.35,
                     label=f"{pct}% CI")
    ax.plot(q_md, M_grid, color="darkblue", linewidth=2.0, label="Median")

    if external_bands:
        for k, ext_b in enumerate(external_bands):
            if "R" not in ext_b or "M_grid" not in ext_b:
                continue
            ext_M = _to_np(ext_b["M_grid"])
            R_lo, R_md, R_hi = (np.array(a, dtype=np.float64)
                                for a in ext_b["R"])
            # Same support threshold as the posterior band, so the two
            # are drawn on the same footing.
            ext_frac = ext_b.get("weight_frac")
            if ext_frac is not None and float(min_weight_frac) > 0:
                ext_mask = _to_np(ext_frac) < float(min_weight_frac)
                R_lo[ext_mask] = np.nan
                R_md[ext_mask] = np.nan
                R_hi[ext_mask] = np.nan
            color = ext_b.get("color", f"C{k+2}")
            lbl   = ext_b.get("label", f"external {k+1}")
            ax.plot(R_lo, ext_M, color=color, linestyle="--",
                    linewidth=1.5, alpha=0.9, zorder=4)
            ax.plot(R_hi, ext_M, color=color, linestyle="--",
                    linewidth=1.5, alpha=0.9, zorder=4)
            ax.plot(R_md, ext_M, color=color, linestyle="-",
                    linewidth=1.8, alpha=0.95, zorder=4, label=lbl)
    if observed_pulsars:
        from matplotlib.lines import Line2D
        _COND_COLOR  = "darkred"
        _CHECK_COLOR = "#2E7D32"
        legend_pulsar_handles = []
        for auto_i, p in enumerate(observed_pulsars, start=1):
            name = p.get("name", "")
            idx  = int(p.get("idx", auto_i))
            tier = p.get("tier", "conditioning")
            is_cond = (tier != "crosscheck")
            color   = _COND_COLOR if is_cond else _CHECK_COLOR
            ax.errorbar(p["R_obs"], p["M_obs"],
                        xerr=float(p["sigma_R"]), yerr=float(p["sigma_M"]),
                        fmt=("o" if is_cond else "D"),
                        color=color, markersize=12,
                        markerfacecolor=(color if is_cond else "white"),
                        markeredgecolor=color, markeredgewidth=1.0,
                        ecolor=color, elinewidth=1.2, capsize=3,
                        capthick=1.2, zorder=20)
            offset = p.get("label_offset")
            if offset is None:
                ax.annotate(str(idx), (p["R_obs"], p["M_obs"]),
                            ha="center", va="center",
                            fontsize=7, fontweight="bold",
                            color=("white" if is_cond else color),
                            zorder=21)
            else:
                ax.annotate(str(idx), (p["R_obs"], p["M_obs"]),
                            textcoords="offset points", xytext=offset,
                            ha="center", va="center",
                            fontsize=7, fontweight="bold", color=color,
                            zorder=21,
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.6, shrinkA=0, shrinkB=1))
            legend_pulsar_handles.append(
                (Line2D([0], [0], marker=("o" if is_cond else "D"),
                        color="none",
                        markerfacecolor=(color if is_cond else "white"),
                        markeredgecolor=color, markeredgewidth=1.0,
                        markersize=11),
                 idx, color, is_cond, name))

    ax.set_xlabel(r"$R$ [km]",        fontsize=12)
    ax.set_ylabel(r"$M / M_\odot$",   fontsize=12)
    ax.grid(True, alpha=0.3)
    from matplotlib.legend_handler import HandlerBase

    class _NumberedMarkerHandler(HandlerBase):
        def __init__(self, idx, color, is_cond):
            super().__init__()
            self._idx = idx
            self._color = color
            self._is_cond = is_cond

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            from matplotlib.lines import Line2D as _L2D
            from matplotlib.text import Text as _Text
            cx = xdescent + width / 2.0
            cy = ydescent + height / 2.0
            marker = _L2D([cx], [cy],
                          marker=("o" if self._is_cond else "D"),
                          markersize=8,
                          markerfacecolor=(self._color if self._is_cond
                                           else "white"),
                          markeredgecolor=self._color,
                          markeredgewidth=1.0,
                          linestyle="none",
                          transform=trans)
            txt = _Text(cx, cy, str(self._idx),
                        ha="center", va="center",
                        fontsize=5, fontweight="bold",
                        color=("white" if self._is_cond else self._color),
                        transform=trans)
            return [marker, txt]

    handles, labels = ax.get_legend_handles_labels()
    handler_map = {}
    pulsar_proxies = []
    pulsar_labels = []
    if observed_pulsars:
        for marker_proxy, idx, color, is_cond, name in legend_pulsar_handles:
            label_str = name if name else f"#{idx}"
            handles.append(marker_proxy)
            labels.append(label_str)
            pulsar_proxies.append(marker_proxy)
            pulsar_labels.append(label_str)
            handler_map[marker_proxy] = _NumberedMarkerHandler(
                idx, color, is_cond)
    ax._pulsar_legend_proxies     = pulsar_proxies
    ax._pulsar_legend_labels      = pulsar_labels
    ax._pulsar_legend_handler_map = dict(handler_map)
    ax.legend(handles, labels,
              handler_map=handler_map,
              fontsize=6, loc="upper left", framealpha=0.9,
              labelspacing=0.7, handlelength=1.6, handletextpad=0.5,
              borderpad=0.3, borderaxespad=0.3)
    if _own_fig:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  M-R band saved to {save_path}")
    return fig, ax


def plot_pt_peak_distribution(diag, save_path=None,
                              title="Speed-of-sound peak posterior"):
    """
    2D weighted scatter of (n_peak, c_s^2_peak) per sample, colored by
    importance weight.  Marker size is constant; the alpha rescaling and
    color make heavy-weight samples visually dominant.

    The annotation in the upper-right reports P_PT, the weighted fraction
    of samples whose c_s^2 dips below `threshold` inside `window`.

    Returns (fig, ax).
    """
    n_peak   = diag["peak_n"]
    cs2_peak = diag["peak_cs2"]
    w        = diag["weights"]
    P_PT     = diag["P_PT"]
    th       = diag["threshold"]
    win      = diag["window"]

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_causal_band(ax, edges=False)
    ax.axhline(1.0 / 3.0, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5, zorder=1)
    if w.max() > 0:
        sizes = 8.0 + 200.0 * (w / w.max())
    else:
        sizes = np.full_like(w, 30.0)

    sc = ax.scatter(n_peak, cs2_peak, s=sizes, c=w, cmap="viridis",
                    alpha=0.65, edgecolor="darkblue", linewidth=0.3,
                    zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Importance weight", fontsize=10)

    note = (f"$P_\\mathrm{{PT}}$ = {P_PT:.3f}\n"
            f"(threshold $c_s^2 < {th:.2f}$\n"
            f"in $n_B/n_0 \\in [{win[0]:.1f}, {win[1]:.1f}]$)")
    ax.text(0.97, 0.97, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlabel(r"$n_B^\mathrm{peak} / n_0$", fontsize=12)
    ax.set_ylabel(r"$c_s^{2,\,\mathrm{peak}}$", fontsize=12)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Peak-distribution plot saved to {save_path}")
    return fig, ax


_TC_TRAIN = "#0072B2"   # blue
_TC_VAL   = "#D55E00"   # vermillion
_TC_LR    = "#999999"   # grey
_TC_BEST  = "#009E73"   # green

_TC_RC = {
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "legend.fontsize":    5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
}


def plot_training_curves(log, save_path=None,
                         columns=("epoch", "train_loss", "val_loss", "lr"),
                         y_label=r"MSE loss (EOS DDPM)",
                         main_ylim=(0.042, 0.10),
                         inset_ylim=(0.042, 0.30),
                         inset_xmax=15,
                         inset_rect=(0.25, 0.41, 0.30, 0.32),
                         warmup_epoch=2,
                         plateau_from=60,
                         show_plateau=True,
                         plateau_label="Convergence\nplateau",
                         plateau_label_y=0.60,
                         panel_labels=("a", "b"),
                         panel_label_offset=(-28, 5),
                         mark_resume=False,
                         figsize=(3.5, 3.5),
                         dpi=300):
    """
    Two-panel training summary for the diffusion run.

    Upper panel: train and validation loss on a log scale, with a star on the
    best (EMA) validation epoch and an inset magnifying the first
    `inset_xmax` epochs at full range.  Lower panel: the learning-rate
    schedule (linear warmup + cosine decay), shaded to the axis.

    A resumed run logs the restart epoch twice -- once before the crash and
    once after -- so rows are de-duplicated on epoch, keeping the last
    occurrence, which is the one the rest of the curve continues from.

    Parameters
    ----------
    log          : path to the epoch-level CSV log written during training, or
                   any column-indexable object (DataFrame, structured array,
                   dict of arrays) carrying `columns`.
    save_path    : if given, write the figure there (format from the suffix);
                   the parent directory is created if it does not exist.
    columns      : (epoch, train, val, lr) column names in the log.
    y_label      : upper-panel y-label -- set it to the loss parameterisation
                   actually trained (v-prediction, eps-prediction, ...).
    main_ylim    : upper-panel limits.  The first epochs deliberately run off
                   the top of the panel; the inset is what resolves them.
    plateau_from : epochs >= this define the shaded convergence band
                   (10th-90th percentile of the validation loss).
    plateau_label, plateau_label_y
                 : text and vertical position (axes fraction) of the band
                   label.  It is centred automatically in the strip to the
                   right of the inset, so moving `inset_rect` moves it too;
                   keep the text short (two lines) to preserve that gap.
    panel_labels : letters drawn at the top-left of each panel for the journal
                   caption ("a", "b"); pass None to omit them.  Positioned by
                   `panel_label_offset`, in points from each panel's top-left
                   corner, so both letters stay mutually aligned and clear of
                   the tick labels at any figure size.
    inset_rect   : [x, y, w, h] of the inset in axes coordinates.  Its tick
                   labels and title add roughly 0.09 below and 0.04 above,
                   which is what has to clear the legend and the curves.
    warmup_epoch : where to mark the end of LR warmup
                   (warmup_steps / steps_per_epoch).
    mark_resume  : draw a vertical line at the resume epoch, taken from the
                   duplicated row in the log.

    Style is applied through rc_context, so the caller's rcParams -- and every
    other figure in the notebook -- are left untouched.

    Returns (fig, (ax_loss, ax_lr)).
    """
    c_ep, c_train, c_val, c_lr = columns

    if isinstance(log, (str, bytes, os.PathLike)):
        rec = np.genfromtxt(log, delimiter=",", names=True)
        col = lambda name: np.asarray(rec[name], dtype=np.float64)
    else:
        col = lambda name: np.asarray(log[name], dtype=np.float64)

    ep_all = col(c_ep)

    _, first_in_reversed = np.unique(ep_all[::-1], return_index=True)
    keep = np.sort(len(ep_all) - 1 - first_in_reversed)

    n_dropped = len(ep_all) - len(keep)
    resume_epoch = None
    if n_dropped:
        dropped = np.setdiff1d(np.arange(len(ep_all)), keep)
        dup = np.unique(ep_all[dropped]).astype(int)
        resume_epoch = int(dup[0])
        print(f"  training log: dropped {n_dropped} duplicated epoch row(s) "
              f"at {dup.tolist()} (kept the post-resume values)")

    epochs = ep_all[keep]
    train  = col(c_train)[keep]
    val    = col(c_val)[keep]
    lr     = col(c_lr)[keep]

    best       = int(np.argmin(val))
    best_epoch = int(epochs[best])
    best_val   = float(val[best])

    with plt.rc_context(_TC_RC):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True,
            gridspec_kw={"hspace": 0.15})

        ax1.plot(epochs, train, color=_TC_TRAIN, lw=1.2,
                 label="Train loss", zorder=3)
        ax1.plot(epochs, val, color=_TC_VAL, lw=1.0, alpha=0.85,
                 label="Val loss", zorder=2)
        ax1.scatter(best_epoch, best_val, marker="*", s=60, color=_TC_BEST,
                    zorder=5, edgecolors="k", linewidths=0.3,
                    label=f"Best EMA val = {best_val:.4f} (ep {best_epoch})")

        ax1.set_ylabel(y_label, fontsize=8)
        ax1.set_yscale("log")
        ax1.set_ylim(*main_ylim)
        ax1.yaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=(4, 5, 6, 8, 10)))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax1.legend(loc="upper right", framealpha=0.9, edgecolor="none")
        ax1.tick_params(axis="x", labelbottom=False)

        if show_plateau:
            late = val[epochs >= plateau_from]
            if late.size:
                lo, hi = np.percentile(late, 10), np.percentile(late, 90)
                ax1.axhspan(lo, hi, color=_TC_TRAIN, alpha=0.06, zorder=0)
                inset_right = inset_rect[0] + inset_rect[2]
                label_x = min(0.95, inset_right + 0.5 * (1.0 - inset_right))
                ax1.annotate(plateau_label,
                             xy=(min(0.90, label_x + 0.06), hi),
                             xycoords=ax1.get_yaxis_transform(),
                             xytext=(label_x, plateau_label_y),
                             textcoords="axes fraction",
                             fontsize=7, color="grey", ha="center",
                             va="center", style="italic", zorder=6,
                             arrowprops=dict(arrowstyle="-|>", color="grey",
                                             lw=0.7))

        if mark_resume and resume_epoch is not None:
            ax1.axvline(resume_epoch, color="k", ls="-.", lw=0.6,
                        alpha=0.4, zorder=1)

        axins = ax1.inset_axes(list(inset_rect))
        axins.plot(epochs, train, color=_TC_TRAIN, lw=0.8)
        axins.plot(epochs, val, color=_TC_VAL, lw=0.8, alpha=0.85)
        axins.set_xlim(0, inset_xmax)
        axins.set_ylim(*inset_ylim)
        axins.set_yscale("log")
        axins.tick_params(labelsize=5)
        axins.set_title("Early epochs", fontsize=5, pad=2)
        axins.yaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=(1, 2, 5)))
        axins.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        axins.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax2.plot(epochs, lr * 1e4, color=_TC_LR, lw=1.2)
        ax2.fill_between(epochs, 0, lr * 1e4, color=_TC_LR, alpha=0.10)
        ax2.set_ylabel(r"LR ($\times 10^{-4}$)", fontsize=8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylim(bottom=-0.02)

        if warmup_epoch:
            ax2.axvline(warmup_epoch, color="k", ls=":", lw=0.6, alpha=0.5)
            ax2.annotate("Warmup", xy=(warmup_epoch + 2, 0.80 * lr.max() * 1e4),
                         fontsize=6, color="grey")
        if panel_labels:
            for ax, letter in zip((ax1, ax2), panel_labels):
                ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                            xytext=panel_label_offset,
                            textcoords="offset points",
                            fontsize=10, fontweight="bold",
                            ha="left", va="bottom")

        fig.align_ylabels([ax1, ax2])

        if save_path:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  Training curves saved to {save_path}")

    return fig, (ax1, ax2)