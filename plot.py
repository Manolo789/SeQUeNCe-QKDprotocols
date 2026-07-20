import os

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR = "data"

# Canonical protocol order (used to keep legend/line order stable).
PROTOCOLS = ["BB84", "B92", "COW"]

# Per-protocol styling for each of the three panels, mirroring the original
# hand-picked colours/dashes.  Keyed by protocol so any subset renders
# consistently.
SKR_STYLE = {
    "BB84": dict(color="blue",  linestyle=(0, (1, 1)),  linewidth=2),
    "B92":  dict(color="green", linestyle=(0, (1, 5)),  linewidth=3),
    "COW":  dict(color="red",   linestyle=(0, (1, 10)), linewidth=4),
}
QBER_STYLE = {
    "BB84": dict(color="orange", linestyle=(0, (5, 1)),  linewidth=2),
    "B92":  dict(color="maroon", linestyle=(0, (5, 5)),  linewidth=2),
    "COW":  dict(color="violet", linestyle=(0, (5, 10)), linewidth=2),
}
RS_STYLE = {
    "BB84": dict(color="grey",  linestyle="solid",                linewidth=3),
    "B92":  dict(color="cyan",  linestyle=(0, (3, 1, 1, 1)),       linewidth=2),
    "COW":  dict(color="black", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2),
}
VIS_STYLE = {
    "":     dict(color="blue",  linestyle=(0, (1, 1)),            linewidth=2),
    "+Eve": dict(color="green", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2),
}

# Human-readable X-axis labels (with units).  Any sweep not listed here falls
# back to its raw column name, so unknown/new sweeps still plot correctly.
X_LABELS = {
    "distance":                   "Distância (d) [m]",
    "keysize":                    "Tamanho de chave (k) [bits]",
    "efficiency":                 "Eficiência do detector",
    "dark_count":                 "Contagem de escuro (dark count) [Hz]",
    "frequency":                  "Frequência da fonte [Hz]",
    "atm_visibility":             "Visibilidade atmosférica [m]",
    "C_n2":                       "Cₙ² — parâmetro de estrutura do índice de refração [m⁻²ᐟ³]",
    "temperature":                "Temperatura [K]",
    "pressure":                   "Pressão [Pa]",
    "wind_speed_perp":            "Velocidade do vento perpendicular [m/s]",
    "height_ag":                  "Altura acima do solo [m]",
    "receiver_radius":            "Raio do receptor [m]",
    "filter_bandwidth":           "Largura de banda do filtro [m]",
    "fov_solid_angle":            "Ângulo sólido do campo de visão (FOV) [sr]",
    "precipitation_rate":         "Taxa de precipitação [m/s]",
    "interferometer_phase_error": "Erro de fase do interferômetro [rad]",
    "eve_intercept_rate":         "Taxa de interceptação da Eve",
    "eve_position":               "Posição da Eve (fração da distância)",
    "hour":                       "Hora do dia [h]",
}

# Force a specific X scale for particular sweeps.  Anything not listed uses the
# automatic heuristic in ``_wants_log_x``.
X_SCALE_OVERRIDE = {
    # e.g. "keysize": "linear",
}

# Fixed operating point of the simulation, mirroring the base values set in
# ``QKD_Extension.py`` (``run_simulation``).  For a given sweep, ``_build_title``
# lists every entry EXCEPT the one being swept, so each figure documents the
# parameters held constant.  Attenuation is intentionally absent: the channel
# loss is now computed dynamically by ``loss.py`` and is no longer a fixed knob.
#
# ``C_n2``, ``fov_solid_angle`` and ``wind_speed_perp`` are computed at runtime
# in QKD_Extension.py; the values below are those base results (hour=12, RH=47%,
# T=298.15 K, wind=3.2 m/s, height=8 m, sensor Ø=1e-4 m, f=0.7004 m).
BASE_PARAMS = {
    "distance":                   "Distância=700 m",
    "keysize":                    "Tamanho de chave=10000 bits",
    "atm_visibility":             "Visib. atm.=10000 m",
    "C_n2":                       "Cₙ²=5,77×10⁻¹⁴ m^(-2/3)",
    "dark_count":                 "Dark count=100 Hz",
    "efficiency":                 "Eficiência=0,65",
    "eve_intercept_rate":         "Taxa intercept. Eve=0,9",
    "eve_position":               "Posição Eve=0,5",
    "filter_bandwidth":           "Larg. banda filtro=1×10⁻⁹ m",
    "fov_solid_angle":            "FOV=1,60×10⁻⁸ sr",
    "frequency":                  "Frequência=8×10⁶ Hz",
    "height_ag":                  "Altura=8 m",
    "interferometer_phase_error": "Erro fase interfer.=0,20 rad",
    "precipitation_rate":         "Precipitação=0 m/s",
    "pressure":                   "Pressão=92700 Pa",
    "receiver_radius":            "Raio receptor=0,103 m",
    "temperature":                "Temperatura=298,15 K",
    "wind_speed_perp":            "Vento perp.=4,34 m/s",
}

# Short sweep names for the legend of the parallel-coordinates grid.
SWEEP_SHORT = {
    "distance":                   "Distância",
    "keysize":                    "Tamanho de chave",
    "hour":                       "Hora do dia",
    "atm_visibility":             "Visib. atmosférica",
    "C_n2":                       "Cₙ²",
    "dark_count":                 "Dark count",
    "efficiency":                 "Eficiência",
    "eve_intercept_rate":         "Taxa intercept. Eve",
    "eve_position":               "Posição Eve",
    "filter_bandwidth":           "Larg. banda filtro",
    "fov_solid_angle":            "FOV",
    "frequency":                  "Frequência",
    "height_ag":                  "Altura",
    "interferometer_phase_error": "Erro fase interfer.",
    "precipitation_rate":         "Precipitação",
    "pressure":                   "Pressão",
    "receiver_radius":            "Raio receptor",
    "temperature":                "Temperatura",
    "wind_speed_perp":            "Vento perp.",
}

# Metrics shown (in this order) in the parallel-coordinates grid, with
# (panel title, Y unit, scale factor applied to the raw value).
METRIC_ORDER = ["R_sk", "QBER", "R_s", "Throughputs", "Latency", "Loss", "Visibility"]
METRIC_INFO = {
    "R_sk":        ("R_sk",            "bits por qubit", 1.0),
    "QBER":        ("QBER",            "%",              100.0),
    "R_s":         ("R_s",             "%",              100.0),
    "Throughputs": ("Throughput",      "qubits/s",       1.0),
    "Latency":     ("Latência",        "s",              1.0),
    "Loss":        ("Perda do canal",  "fração",         1.0),
    "Visibility":  ("Visibilidade",    "",               1.0),
}


# ═══════════════════════════════════════════════════════════════════════
#  Small helpers
# ═══════════════════════════════════════════════════════════════════════
def safe_log10(values) -> np.ndarray:
    """log10 that maps non-positive entries to NaN (so they are skipped)."""
    arr = np.array(values, dtype=float)
    arr[arr <= 0] = np.nan
    return np.log10(arr)


def _label(sweep_var: str) -> str:
    return X_LABELS.get(sweep_var, sweep_var)


def _short_sweep(sweep_var: str) -> str:
    return SWEEP_SHORT.get(sweep_var, sweep_var)


def _normalize_x(values) -> np.ndarray:
    """Map a sweep's X values to [0, 1] (fraction of the swept range).

    Uses log-space when the sweep spans many orders of magnitude and is all
    positive (same heuristic as the X-axis scale), so log-sampled sweeps
    (C_n2, frequency, ...) get evenly spaced normalised points.
    """
    v = np.asarray(values, dtype=float)
    if _wants_log_x(v):
        with np.errstate(invalid="ignore"):
            v = np.where(v > 0, np.log10(v), np.nan)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi == lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def _wants_log_x(values) -> bool:
    """Use a log X-axis when the sweep spans many orders of magnitude.

    Only kicks in when every value is strictly positive (a log axis would drop
    a 0, e.g. ``interferometer_phase_error`` starts at 0) and the ratio between
    the largest and smallest value is at least 100.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.any(v <= 0):
        return False
    return (v.max() / v.min()) >= 100


def _x_scale(sweep_var: str, values) -> str:
    return X_SCALE_OVERRIDE.get(
        sweep_var, "log" if _wants_log_x(values) else "linear"
    )


def _present_protocols(df: pd.DataFrame, suffix: str) -> list:
    """Protocols that have data for this scenario (``suffix`` = "" or "+Eve")."""
    return [p for p in PROTOCOLS if f"R_sk-{p}{suffix}" in df.columns]


def _legenda(metric: str, proto: str, suffix: str) -> str:
    """Rótulo de legenda em português: 'R_sk do BB84', 'QBER do B92 com Eve'."""
    base = f"{metric} do {proto}"
    return base + (" com Eve" if suffix else "")


def _series(df: pd.DataFrame, metric: str, proto: str, suffix: str):
    """Return column values, or None if the column is absent/all-empty."""
    col = f"{metric}-{proto}{suffix}"
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(vals)):
        return None
    return vals


# ═══════════════════════════════════════════════════════════════════════
#  Single-scenario figure (R_sk + QBER on top, R_s on the bottom)
# ═══════════════════════════════════════════════════════════════════════
def plot_scenario(df, sweep_var, suffix, title, out_path, x_scale="linear"):
    """Draw one scenario (ideal or +Eve) for whatever protocols are present."""
    protocols = _present_protocols(df, suffix)
    if not protocols:
        return  # nothing to draw for this scenario

    x = np.asarray(df[sweep_var], dtype=float)
    fig, (ax_top, ax_bot) = plt.subplots(2, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=10)

    handles = []

    # — Top-left: secret-key rate (log scale) —
    for p in protocols:
        y = _series(df, "R_sk", p, suffix)
        if y is None:
            continue
        (ln,) = ax_top.plot(x, safe_log10(y), label=_legenda("R_sk", p, suffix), **SKR_STYLE[p])
        handles.append(ln)
    ax_top.set_ylabel("log₁₀ Taxa de chave secreta (R_sk)\n[bits por qubit enviado]")
    ax_top.grid(True)

    # — Top-right (twin): QBER —
    ax_q = ax_top.twinx()
    for p in protocols:
        y = _series(df, "QBER", p, suffix)
        if y is None:
            continue
        (ln,) = ax_q.plot(x, y * 100, label=_legenda("QBER", p, suffix), **QBER_STYLE[p])
        handles.append(ln)
    ax_q.set_ylabel("QBER [%]")

    # — Bottom: useful-bit rate —
    for p in protocols:
        y = _series(df, "R_s", p, suffix)
        if y is None:
            continue
        (ln,) = ax_bot.plot(x, y * 100, label=_legenda("R_s", p, suffix), **RS_STYLE[p])
        handles.append(ln)
    ax_bot.set_xlabel(_label(sweep_var))
    ax_bot.set_ylabel("R_s - Taxa de bits úteis [%]")
    ax_bot.set_xscale(x_scale)
    ax_bot.grid(True)

    labels = [h.get_label() for h in handles]
    ax_bot.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  fancybox=True, shadow=True, ncol=5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  Visibility figure (COW only)
# ═══════════════════════════════════════════════════════════════════════
def plot_visibility(df, sweep_var, title, out_path, x_scale="linear"):
    """Draw the COW visibility (ideal and/or +Eve) if present."""
    have = [s for s in ("", "+Eve") if f"Visibility-COW{s}" in df.columns]
    if not have:
        return

    x = np.asarray(df[sweep_var], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=10)

    handles = []
    for suffix in have:
        y = pd.to_numeric(df[f"Visibility-COW{suffix}"], errors="coerce").to_numpy(float)
        if not np.any(np.isfinite(y)):
            continue
        (ln,) = ax.plot(x, y, label=_legenda("V", "COW", suffix), **VIS_STYLE[suffix])
        handles.append(ln)
    if not handles:
        plt.close()
        return

    ax.set_xlabel(_label(sweep_var))
    ax.set_ylabel("V - Visibilidade [%]")
    ax.set_xscale(x_scale)
    ax.grid(True)
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  All figures for a single sweep file
# ═══════════════════════════════════════════════════════════════════════
def plot_sweep(df, filename, title=None):
    """Generate every applicable figure for one sweep CSV."""
    sweep_var = df.columns[0]
    x_scale = _x_scale(sweep_var, df[sweep_var])
    
    
    if title is None:
        title = ""
    # Ideal scenario (columns with no +Eve suffix)
    plot_scenario(df, sweep_var, suffix="", title=title,
                  out_path=f"{DATA_DIR}/{filename}_graph-ideal_scenario.png",
                  x_scale=x_scale)

    # Eavesdropper scenario (+Eve columns)
    plot_scenario(df, sweep_var, suffix="+Eve", title=title,
                  out_path=f"{DATA_DIR}/{filename}_graph-Eve_scenario.png",
                  x_scale=x_scale)

    # COW visibility
    plot_visibility(df, sweep_var, title=title,
                    out_path=f"{DATA_DIR}/{filename}_graph-visibility.png",
                    x_scale=x_scale)


# ═══════════════════════════════════════════════════════════════════════
#  Side-by-side comparison of two sweeps (e.g. distance vs. key size)
# ═══════════════════════════════════════════════════════════════════════
def plot_dual_graph(df_left, df_right, suffix, title, filename,
                    subtitle_left, subtitle_right):
    """Two sweeps side by side for one scenario (``suffix`` = "" or "+Eve")."""
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fontsize = 15
    fontsize_legend = 16
    fig.suptitle(title, fontsize=fontsize, y=0.925)

    def _draw_column(ax_top, ax_bot, df, subtitle,
                     show_left_ylabel=True, show_right_ylabel=True):
        sweep_var = df.columns[0]
        x = np.asarray(df[sweep_var], dtype=float)
        protocols = _present_protocols(df, suffix)
        ax_top.set_title(subtitle, fontsize=fontsize)

        # Top: R_sk (left) + QBER (right twin)
        for p in protocols:
            y = _series(df, "R_sk", p, suffix)
            if y is not None:
                ax_top.plot(x, safe_log10(y), label=_legenda("R_sk", p, suffix), **SKR_STYLE[p])
        ax_top.set_ylabel(
            "log₁₀ Taxa de chave secreta (R_sk)\n[bits por qubit enviado]"
            if show_left_ylabel else "", fontsize=fontsize)
        ax_top.grid(True)

        ax_q = ax_top.twinx()
        for p in protocols:
            y = _series(df, "QBER", p, suffix)
            if y is not None:
                ax_q.plot(x, y * 100, label=_legenda("QBER", p, suffix), **QBER_STYLE[p])
        ax_q.set_ylabel("QBER [%]" if show_right_ylabel else "", fontsize=fontsize)

        # Bottom: R_s
        for p in protocols:
            y = _series(df, "R_s", p, suffix)
            if y is not None:
                ax_bot.plot(x, y * 100, label=_legenda("R_s", p, suffix), **RS_STYLE[p])
        ax_bot.set_xlabel(_label(sweep_var), fontsize=fontsize)
        ax_bot.set_ylabel("R_s - Taxa de bits úteis [%]" if show_left_ylabel else "",
                          fontsize=fontsize)
        ax_bot.set_xscale(_x_scale(sweep_var, df[sweep_var]))
        ax_bot.grid(True)

        return ax_top.get_lines() + ax_q.get_lines() + ax_bot.get_lines()

    lines_left = _draw_column(axes[0, 0], axes[1, 0], df_left, subtitle_left,
                              show_left_ylabel=True, show_right_ylabel=False)
    _draw_column(axes[0, 1], axes[1, 1], df_right, subtitle_right,
                 show_left_ylabel=False, show_right_ylabel=True)

    labels = [l.get_label() for l in lines_left]
    fig.legend(lines_left, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               fancybox=True, shadow=True, ncol=5, fontsize=fontsize_legend, markerscale=3)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(f"{DATA_DIR}/{filename}_graph-dual.png", dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  Parallel-coordinates grid: metrics × sweeps for one protocol/scenario
# ═══════════════════════════════════════════════════════════════════════
def _amplitude(y) -> float:
    """Sensitivity of a curve = (max - min), ignoring NaN."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    return float(y.max() - y.min())


def plot_parallel_grid(sweep_dfs, protocol, suffix, title, filename,
                       ncols=3, highlight_top=None):
    """Grid of metric panels; in each panel one curve per sweep.

    For a fixed ``protocol`` and scenario (``suffix`` = "" ideal or "+Eve"),
    every metric that exists gets its own subplot.  Inside a subplot, each sweep
    that varied is drawn as a curve of the metric versus its **normalised**
    parameter (X mapped to [0, 1] = fraction of the swept range), so responses
    with different physical units/ranges can be compared by shape on one axis.

    Args:
        sweep_dfs: ordered mapping ``{sweep_name: DataFrame}``.
        protocol:  "BB84", "B92" or "COW".
        suffix:    "" (ideal) or "+Eve".
        title, filename: figure super-title and output stem (data/{filename}.png).
        ncols: number of columns in the panel grid.
        highlight_top: if ``None`` (default) every sweep is drawn with its own
            colour and appears in the legend (original look).  If an integer N,
            then in each panel the N most influential sweeps FOR THAT METRIC are
            highlighted in colour (fixed colour per sweep, reused across panels)
            and the rest are drawn as darker-grey background curves.  The legend
            lists the union of highlighted sweeps plus "Outros (pouco influentes)".

    Returns:
        True if a figure was produced, False if there was no data.
    """
    # Metrics that have at least one column for this protocol/scenario.
    metrics = [m for m in METRIC_ORDER
               if any(f"{m}-{protocol}{suffix}" in df.columns for df in sweep_dfs.values())]
    if not metrics:
        return False

    # Sweeps that carry data for this protocol/scenario (fixes colour order).
    sweeps = [s for s, df in sweep_dfs.items()
              if any(f"{m}-{protocol}{suffix}" in df.columns for m in metrics)]
    if not sweeps:
        return False

    highlight_mode = highlight_top is not None and highlight_top < len(sweeps)
    if highlight_mode:
        # Top-N most influential sweeps PER METRIC (by amplitude of that metric).
        top_by_metric = {}
        for m in metrics:
            _, _, scale = METRIC_INFO[m]
            amps = []
            for s in sweeps:
                y = _series(sweep_dfs[s], m, protocol, suffix)
                if y is not None:
                    amps.append((_amplitude(y * scale), s))
            amps.sort(reverse=True, key=lambda t: t[0])
            top_by_metric[m] = {s for _, s in amps[:highlight_top]}
        # Union of all highlighted sweeps (ordered by metric order, then rank),
        # each gets a fixed colour reused in every panel.
        union = []
        for m in metrics:
            for s in sweeps:                       # keep a stable order
                if s in top_by_metric[m] and s not in union:
                    union.append(s)
        cmap = plt.get_cmap("tab20")
        colors = {s: cmap(i % 20) for i, s in enumerate(union)}
        legend_sweeps = union
        GREY = "0.6"          # a bit darker than before for the faded curves
    else:
        cmap = plt.get_cmap("tab20")
        colors = {s: cmap(i % 20) for i, s in enumerate(sweeps)}
        legend_sweeps = sweeps

    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=13)

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        name, unit, scale = METRIC_INFO[metric]

        for s in sweeps:
            df = sweep_dfs[s]
            y = _series(df, metric, protocol, suffix)
            if y is None:
                continue
            xn = _normalize_x(df[df.columns[0]])
            order = np.argsort(xn)
            xv, yv = xn[order], (y * scale)[order]
            # Highlight only if this sweep is in the TOP-N of THIS metric.
            if highlight_mode and s not in top_by_metric[metric]:
                ax.plot(xv, yv, color=GREY, linewidth=0.9, alpha=0.7, zorder=1)
            else:
                ax.plot(xv, yv, color=colors[s], linewidth=1.9 if highlight_mode else 1.6,
                        marker="o", markersize=2.5, zorder=3)

        ax.set_title(name, fontsize=11)
        ax.set_ylabel(f"{name} [{unit}]" if unit else name, fontsize=9)
        ax.grid(True, alpha=0.4)
        if r == nrows - 1:
            ax.set_xlabel("Fração da faixa do sweep (normalizada)", fontsize=9)

    # Hide any leftover empty panels.
    for idx in range(len(metrics), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    # Single legend = union of highlighted sweeps (fixed colours) + "Outros".
    handles = [mpl.lines.Line2D([], [], color=colors[s], linewidth=2,
                                marker="o", markersize=4) for s in legend_sweeps]
    labels = [_short_sweep(s) for s in legend_sweeps]
    if highlight_mode:
        handles.append(mpl.lines.Line2D([], [], color=GREY, linewidth=1.2, alpha=0.7))
        labels.append("Outros (pouco influentes)")
    fig.legend(handles, labels, loc="lower center",
               ncol=min(6, len(labels)), fontsize=9,
               fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"{DATA_DIR}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Tornado: sensitivity ranking (option B)
# ═══════════════════════════════════════════════════════════════════════
def plot_tornado_grid(sweep_dfs, protocol, suffix, title, filename, ncols=3):
    """One panel per metric; horizontal bars = each sweep's sensitivity
    (amplitude max-min of the metric), sorted so the most influential is on top.
    """
    metrics = [m for m in METRIC_ORDER
               if any(f"{m}-{protocol}{suffix}" in df.columns for df in sweep_dfs.values())]
    if not metrics:
        return False
    sweeps = [s for s, df in sweep_dfs.items()
              if any(f"{m}-{protocol}{suffix}" in df.columns for m in metrics)]
    if not sweeps:
        return False

    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=13)

    bar_cmap = plt.get_cmap("viridis")
    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        name, unit, scale = METRIC_INFO[metric]

        ranked = []
        for s in sweeps:
            y = _series(sweep_dfs[s], metric, protocol, suffix)
            if y is None:
                continue
            ranked.append((_amplitude(y * scale), s))
        ranked.sort(key=lambda t: t[0])          # smallest at bottom, largest on top
        vals = [a for a, _ in ranked]
        labels = [_short_sweep(s) for _, s in ranked]
        cols = bar_cmap(np.linspace(0.15, 0.9, len(vals))) if vals else []

        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(f"Amplitude (máx−mín) [{unit}]" if unit else "Amplitude (máx−mín)",
                      fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    for idx in range(len(metrics), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{DATA_DIR}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Heatmap: sweeps × metrics sensitivity (option C)
# ═══════════════════════════════════════════════════════════════════════
def plot_sensitivity_heatmap(sweep_dfs, protocol, suffix, title, filename):
    """Matrix (sweeps × metrics); colour = amplitude normalised per metric
    (each column scaled to [0, 1]); each cell annotated with the value.
    """
    metrics = [m for m in METRIC_ORDER
               if any(f"{m}-{protocol}{suffix}" in df.columns for df in sweep_dfs.values())]
    if not metrics:
        return False
    sweeps = [s for s, df in sweep_dfs.items()
              if any(f"{m}-{protocol}{suffix}" in df.columns for m in metrics)]
    if not sweeps:
        return False

    mat = np.full((len(sweeps), len(metrics)), np.nan)
    for j, metric in enumerate(metrics):
        _, _, scale = METRIC_INFO[metric]
        col_amps = np.array([
            _amplitude(_series(sweep_dfs[s], metric, protocol, suffix) * scale)
            if _series(sweep_dfs[s], metric, protocol, suffix) is not None else np.nan
            for s in sweeps
        ])
        mx = np.nanmax(col_amps) if np.any(np.isfinite(col_amps)) else 0.0
        mat[:, j] = col_amps / mx if mx > 0 else col_amps

    fig, ax = plt.subplots(figsize=(1.15 * len(metrics) + 3, 0.42 * len(sweeps) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_INFO[m][0] for m in metrics],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(sweeps)))
    ax.set_yticklabels([_short_sweep(s) for s in sweeps], fontsize=8)
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Amplitude relativa (0–1, por coluna)", fontsize=8)

    for i in range(len(sweeps)):
        for j in range(len(metrics)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[i, j] < 0.6 else "black")

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════
def _stem_from_path(path: str) -> str:
    """metrics_variable-distance.csv -> distance ; metrics_scenario-hour.csv -> hour."""
    name = os.path.splitext(os.path.basename(path))[0]
    for prefix in ("metrics_variable-", "metrics_scenario-", "metrics_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# Sweeps to plot, in the desired order.  ``hour`` comes from a
# ``metrics_scenario-*`` file; every other one from ``metrics_variable-*``.
# NOTE: the CSV/column names produced by the simulator are ``dark_count``
# (not ``dark_counts``) and ``height_ag`` (not ``height``).
SWEEPS = [
    "distance",
    "keysize",
    "hour",
    "atm_visibility",
    "C_n2",
    "dark_count",
    "efficiency",
    "eve_intercept_rate",
    "eve_position",
    "filter_bandwidth",
    "fov_solid_angle",
    "frequency",
    "height_ag",
    "interferometer_phase_error",
    "precipitation_rate",
    "pressure",
    "receiver_radius",
    "temperature",
    "wind_speed_perp",
]


def _resolve_csv(sweep: str):
    """Return the CSV path for a sweep name, or None if no file exists."""
    for template in (f"metrics_variable-{sweep}.csv", f"metrics_scenario-{sweep}.csv"):
        path = os.path.join(DATA_DIR, template)
        if os.path.exists(path):
            return path
    return None


def _build_title(sweep_var, per_line: int = 5) -> str:
    """Title listing every fixed parameter except the one being swept.

    Values come from ``BASE_PARAMS`` (the ``QKD_Extension.py`` operating point).
    Attenuation is not shown — channel loss is computed dynamically by loss.py.
    """
    items = [txt for name, txt in BASE_PARAMS.items() if name != sweep_var]
    # Wrap into lines of ``per_line`` items, separated by " · ".
    lines = [" · ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "Parâmetros fixos: " + "\n".join(lines)


def main():
    # Fixed operating point (matches QKD_Extension.py).  Attenuation is no longer
    # a parameter here: channel loss is computed dynamically by loss.py.
    distance = 700     # m
    keysize = 10000    # bits

    # Plot the metrics as a function of each sweep, one figure set per sweep,
    # in the order defined by SWEEPS.
    dfs = {}          # keyed by the actual variable name (first CSV column)
    sweep_dfs = {}    # keyed by the sweep name (SWEEPS order) — for the grid
    for sweep in SWEEPS:
        path = _resolve_csv(sweep)
        if path is None:
            print(f"[--] '{sweep}': CSV não encontrado em '{DATA_DIR}/', pulando.")
            continue
        df = pd.read_csv(path)
        sweep_var = df.columns[0]          # actual variable name (first column)
        dfs[sweep_var] = df
        sweep_dfs[sweep] = df
        stem = _stem_from_path(path)
        #title = _build_title(sweep_var)
        #plot_sweep(df, filename=stem, title=title)
        plot_sweep(df, filename=stem)
        print(f"[ok] {sweep}  ({os.path.basename(path)})")

    # Special side-by-side comparison: distance vs. key size (if both exist).
    if "distance" in dfs and "keysize" in dfs:
        df_d, df_k = dfs["distance"], dfs["keysize"]
        subtitle_d = f"Variação da distância - Tamanho da Chave={keysize} bits"
        subtitle_k = f"Variação do tamanho da chave - Distância={distance} m"

        plot_dual_graph(df_d, df_k, suffix="", title="Cenário ideal (sem espião)",
                        filename="ideal_scenario",
                        subtitle_left=subtitle_d, subtitle_right=subtitle_k)
        plot_dual_graph(df_d, df_k, suffix="+Eve", title="Cenário com Eve",
                        filename="eve_scenario",
                        subtitle_left=subtitle_d, subtitle_right=subtitle_k)
        print("[ok] comparação dupla distance × keysize")

    # Comparison figures per (protocol, scenario):
    #   A) parallel-coordinates grid (top-N highlighted, others faded)
    #   B) tornado (sensitivity ranking)
    #   C) sensitivity heatmap (sweeps × metrics)
    scenarios = {"": ("ideal", "Cenário ideal (sem espião)"),
                 "+Eve": ("Eve", "Cenário com Eve")}
    highlight_top = 6   # None -> original look (all sweeps coloured/legended)
    for proto in PROTOCOLS:
        for suffix, (tag, scen_label) in scenarios.items():
            head = f"{proto} — {scen_label}"

            # A) parallel-coordinates grid
            titleA = head + "\nMétricas × sweeps  (eixo X normalizado por sweep)"
            if plot_parallel_grid(sweep_dfs, proto, suffix, title=titleA,
                                  filename=f"parallel_{proto}_{tag}",
                                  highlight_top=highlight_top):
                print(f"[ok] A) grade de sweeps: {proto} ({tag})")

            # B) tornado
            titleB = head + "\nTornado: sensibilidade de cada métrica a cada sweep"
            if plot_tornado_grid(sweep_dfs, proto, suffix, title=titleB,
                                 filename=f"tornado_{proto}_{tag}"):
                print(f"[ok] B) tornado: {proto} ({tag})")

            # C) heatmap
            titleC = head + " · Sensibilidade normalizada (por métrica)"
            if plot_sensitivity_heatmap(sweep_dfs, proto, suffix, title=titleC,
                                        filename=f"heatmap_{proto}_{tag}"):
                print(f"[ok] C) heatmap: {proto} ({tag})")


if __name__ == "__main__":
    main()
