"""Matplotlib (and optional Plotly) plotting integration for sounio epistemic types.

All functions accept an optional ``ax`` (matplotlib Axes).  If None, a new
figure + axes pair is created.  Every function returns the axes object so
callers can further customise the plot.

Plotly variants are gated behind a try/except import and raise
``ImportError`` when plotly is not installed.

Catppuccin Mocha dark-theme palette is used throughout:
  background : #1e1e2e
  text/foreground : #cdd6f4
  green  : #a6e3a1
  yellow : #f9e2af
  red    : #f38ba8
  blue   : #89b4fa
  mauve  : #cba6f7
  peach  : #fab387
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    _HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    go = None  # type: ignore[assignment]

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

from ..knowledge import Knowledge
from ..types import PKParameters

# ---------------------------------------------------------------------------
# Catppuccin Mocha colour constants
# ---------------------------------------------------------------------------

_BG = "#1e1e2e"
_FG = "#cdd6f4"
_GREEN = "#a6e3a1"
_YELLOW = "#f9e2af"
_RED = "#f38ba8"
_BLUE = "#89b4fa"
_MAUVE = "#cba6f7"
_PEACH = "#fab387"

_PALETTE = [_GREEN, _YELLOW, _RED, _BLUE, _MAUVE, _PEACH]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_matplotlib() -> None:
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for this function: pip install matplotlib")


def _require_plotly() -> None:
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for plotly variants: pip install plotly")


def _make_ax(ax: Optional[object]) -> tuple:
    """Return (fig, ax), creating a new figure if *ax* is None."""
    if ax is None:
        fig, ax = plt.subplots(facecolor=_BG)
    else:
        fig = ax.get_figure()
    _style_ax(ax)
    return fig, ax


def _style_ax(ax) -> None:
    """Apply Catppuccin Mocha dark theme to a matplotlib Axes."""
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_FG)
        spine.set_alpha(0.4)


def _extract_values_epsilons(
    items: Sequence[Union[Knowledge, float, int]],
) -> tuple:
    """Return (values, epsilons) lists from a sequence of Knowledge or numerics."""
    values: List[float] = []
    epsilons: List[float] = []
    for item in items:
        if isinstance(item, Knowledge):
            values.append(item.value)
            epsilons.append(item.epsilon)
        else:
            values.append(float(item))
            epsilons.append(0.0)
    return values, epsilons


# ---------------------------------------------------------------------------
# 1. Bar chart with error bars
# ---------------------------------------------------------------------------


def plot_knowledge_bar(
    knowledges: Sequence[Union[Knowledge, float, int]],
    labels: Sequence[str],
    ax=None,
):
    """Bar chart with error bars derived from Knowledge.epsilon (k=2, 95 % CI).

    Parameters
    ----------
    knowledges : sequence of Knowledge or float
        Central values and uncertainties.
    labels : sequence of str
        Bar labels (must match length of *knowledges*).
    ax : matplotlib Axes, optional
        Target axes.  Created if None.

    Returns
    -------
    ax : matplotlib Axes
    """
    _require_matplotlib()
    if len(knowledges) != len(labels):
        raise ValueError("knowledges and labels must have the same length")

    values, epsilons = _extract_values_epsilons(knowledges)
    # k=2 coverage factor for ~95 % CI
    yerr = [2.0 * e for e in epsilons]

    fig, ax = _make_ax(ax)
    x = list(range(len(values)))
    colors = [_PALETTE[i % len(_PALETTE)] for i in x]

    ax.bar(
        x,
        values,
        yerr=yerr,
        color=colors,
        error_kw={"ecolor": _FG, "capsize": 4, "alpha": 0.8},
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), color=_FG)
    ax.set_ylabel("Value ± 2σ", color=_FG)
    return ax


# ---------------------------------------------------------------------------
# 2. Concentration–time curve with uncertainty band
# ---------------------------------------------------------------------------


def plot_concentration_time(
    times: Sequence[float],
    concentrations: Sequence[Union[Knowledge, float, int]],
    ax=None,
):
    """Line plot with ±2σ shaded uncertainty band.

    Parameters
    ----------
    times : sequence of float
        Time points.
    concentrations : sequence of Knowledge or float
        Concentration at each time point.  Knowledge.epsilon is used as σ.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    _require_matplotlib()

    # Support UncertainArray duck-typing (has .values / .epsilons attributes)
    if hasattr(concentrations, "values") and hasattr(concentrations, "epsilons"):
        values = list(concentrations.values)
        epsilons = list(concentrations.epsilons)
    else:
        values, epsilons = _extract_values_epsilons(concentrations)

    t = list(times)
    upper = [v + 2.0 * e for v, e in zip(values, epsilons)]
    lower = [v - 2.0 * e for v, e in zip(values, epsilons)]

    fig, ax = _make_ax(ax)
    ax.plot(t, values, color=_GREEN, linewidth=2, label="C(t)")
    ax.fill_between(t, lower, upper, color=_GREEN, alpha=0.25, label="±2σ")
    ax.set_xlabel("Time", color=_FG)
    ax.set_ylabel("Concentration", color=_FG)
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor=_FG)
    return ax


# ---------------------------------------------------------------------------
# 3. Funnel plot (meta-analysis / clinical trial)
# ---------------------------------------------------------------------------


def plot_funnel(
    results: Sequence[Dict],
    ax=None,
):
    """Funnel plot: x = effect size, y = 1/SE (precision).

    Parameters
    ----------
    results : sequence of dict
        Each dict must contain:
        - ``"effect"`` : Knowledge or float — effect size estimate
        - ``"se"`` : float — standard error (must be > 0)
        Optional key: ``"label"`` for annotation.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    _require_matplotlib()
    fig, ax = _make_ax(ax)

    for i, r in enumerate(results):
        effect = r["effect"]
        se = float(r.get("se", 1.0))
        label = r.get("label", "")
        if se <= 0:
            se = 1e-9
        precision = 1.0 / se
        eff_val = effect.value if isinstance(effect, Knowledge) else float(effect)
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter([eff_val], [precision], color=color, zorder=3)
        if label:
            ax.annotate(
                label,
                (eff_val, precision),
                textcoords="offset points",
                xytext=(5, 3),
                color=_FG,
                fontsize=8,
            )

    # Funnel lines — only if there are results
    if results:
        all_precisions = []
        all_effects = []
        for r in results:
            se = float(r.get("se", 1.0))
            if se <= 0:
                se = 1e-9
            all_precisions.append(1.0 / se)
            effect = r["effect"]
            all_effects.append(effect.value if isinstance(effect, Knowledge) else float(effect))
        pooled = sum(all_effects) / len(all_effects)
        max_prec = max(all_precisions) if all_precisions else 1.0
        prec_range = [0, max_prec * 1.1]
        ax.plot(
            [pooled - 2.0 * (1.0 / p if p > 0 else 0) for p in prec_range],
            prec_range,
            color=_YELLOW,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.plot(
            [pooled + 2.0 * (1.0 / p if p > 0 else 0) for p in prec_range],
            prec_range,
            color=_YELLOW,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.axvline(pooled, color=_RED, linewidth=1, linestyle=":", alpha=0.8)

    ax.set_xlabel("Effect size", color=_FG)
    ax.set_ylabel("Precision (1/SE)", color=_FG)
    ax.set_title("Funnel plot", color=_FG)
    return ax


# ---------------------------------------------------------------------------
# 4. PK profile — one-compartment model with GUM uncertainty band
# ---------------------------------------------------------------------------


def plot_pk_profile(
    pk_params: PKParameters,
    dose: Union[Knowledge, float],
    t_max: float = 24.0,
    n_points: int = 100,
    ax=None,
):
    """One-compartment concentration–time profile with GUM uncertainty band.

    C(t) = (F · D / Vd) · exp(−CL · t / Vd)

    GUM uncertainty propagation is performed analytically at each time point.

    Parameters
    ----------
    pk_params : PKParameters
        Fitted PK parameters (clearance, volume_dist, bioavailability).
    dose : Knowledge or float
        Administered dose.
    t_max : float
        Maximum time (same units as clearance denominator, default 24 h).
    n_points : int
        Number of time points to evaluate.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    _require_matplotlib()

    F = pk_params.bioavailability
    CL = pk_params.clearance
    Vd = pk_params.volume_dist

    if isinstance(dose, Knowledge):
        D = dose
    else:
        D = Knowledge(float(dose), 0.0, "dose")

    # Central estimates
    F_v, F_e = F.value, F.epsilon
    CL_v, CL_e = CL.value, CL.epsilon
    Vd_v, Vd_e = Vd.value, Vd.epsilon
    D_v, D_e = D.value, D.epsilon

    step = t_max / max(n_points - 1, 1)
    t_vals = [i * step for i in range(n_points)]
    c_vals: List[float] = []
    c_upper: List[float] = []
    c_lower: List[float] = []

    for t in t_vals:
        if Vd_v == 0.0:
            c = 0.0
            eps = 0.0
        else:
            exp_factor = math.exp(-CL_v * t / Vd_v)
            c = (F_v * D_v / Vd_v) * exp_factor

            # Partial derivatives for GUM propagation
            # dC/dF = (D/Vd)*exp
            # dC/dD = (F/Vd)*exp
            # dC/dVd = -F*D/Vd^2 * exp + F*D/Vd * (CL*t/Vd^2)*exp
            #        = (F*D*exp/Vd^2) * (CL*t/Vd - 1)
            # dC/dCL = F*D/Vd * exp * (-t/Vd)
            dC_dF = (D_v / Vd_v) * exp_factor if Vd_v != 0.0 else 0.0
            dC_dD = (F_v / Vd_v) * exp_factor if Vd_v != 0.0 else 0.0
            dC_dVd = ((F_v * D_v * exp_factor) / (Vd_v ** 2)) * (CL_v * t / Vd_v - 1.0) if Vd_v != 0.0 else 0.0
            dC_dCL = (F_v * D_v / Vd_v) * exp_factor * (-t / Vd_v) if Vd_v != 0.0 else 0.0

            eps = math.sqrt(
                (dC_dF * F_e) ** 2
                + (dC_dD * D_e) ** 2
                + (dC_dVd * Vd_e) ** 2
                + (dC_dCL * CL_e) ** 2
            )

        c_vals.append(c)
        c_upper.append(c + 2.0 * eps)
        c_lower.append(max(0.0, c - 2.0 * eps))

    fig, ax = _make_ax(ax)
    ax.plot(t_vals, c_vals, color=_BLUE, linewidth=2, label="C(t)")
    ax.fill_between(t_vals, c_lower, c_upper, color=_BLUE, alpha=0.25, label="±2σ GUM")
    ax.set_xlabel("Time (h)", color=_FG)
    ax.set_ylabel("Concentration", color=_FG)
    ax.set_title("PK Profile (one-compartment)", color=_FG)
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor=_FG)
    return ax


# ---------------------------------------------------------------------------
# 5. Scatter plot with bivariate error bars
# ---------------------------------------------------------------------------


def plot_knowledge_scatter(
    x_knowledges: Sequence[Union[Knowledge, float, int]],
    y_knowledges: Sequence[Union[Knowledge, float, int]],
    labels: Optional[Sequence[str]] = None,
    ax=None,
):
    """Scatter plot with error bars in both x and y directions.

    Parameters
    ----------
    x_knowledges : sequence of Knowledge or float
    y_knowledges : sequence of Knowledge or float
    labels : sequence of str, optional
        Point labels for annotation.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    _require_matplotlib()
    if len(x_knowledges) != len(y_knowledges):
        raise ValueError("x_knowledges and y_knowledges must have the same length")

    x_vals, x_eps = _extract_values_epsilons(x_knowledges)
    y_vals, y_eps = _extract_values_epsilons(y_knowledges)
    xerr = [2.0 * e for e in x_eps]
    yerr = [2.0 * e for e in y_eps]

    fig, ax = _make_ax(ax)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(x_vals))]

    ax.errorbar(
        x_vals,
        y_vals,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        color=_MAUVE,
        ecolor=_FG,
        capsize=3,
        alpha=0.85,
        markersize=6,
    )

    if labels:
        for xv, yv, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(
                lbl,
                (xv, yv),
                textcoords="offset points",
                xytext=(5, 3),
                color=_FG,
                fontsize=8,
            )

    ax.set_xlabel("X (± 2σ)", color=_FG)
    ax.set_ylabel("Y (± 2σ)", color=_FG)
    return ax


# ---------------------------------------------------------------------------
# 6. Epistemic heatmap — relative uncertainty across variables
# ---------------------------------------------------------------------------


def plot_epistemic_heatmap(
    data_dict: Dict[str, Union[Knowledge, float, int]],
    ax=None,
):
    """Heatmap of relative uncertainty (ε / |value|) for a set of variables.

    Parameters
    ----------
    data_dict : dict[str, Knowledge or float]
        Mapping of variable name → value.  For float values the relative
        uncertainty is 0.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    _require_matplotlib()

    names = list(data_dict.keys())
    rel_uncertainties: List[float] = []
    for v in data_dict.values():
        if isinstance(v, Knowledge):
            rel_uncertainties.append(v.relative_uncertainty)
        else:
            rel_uncertainties.append(0.0)

    # Shape as a single-row heatmap
    n = len(names)
    data_2d = [rel_uncertainties]  # 1 × n

    fig, ax = _make_ax(ax)

    import matplotlib.colors as mcolors

    # Custom colormap: dark green (low) → yellow → red (high)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mocha_uncertainty",
        [_GREEN, _YELLOW, _RED],
    )

    im = ax.imshow(data_2d, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(names, rotation=45, ha="right", color=_FG)
    ax.set_yticks([])

    # Annotate cells
    for j, ru in enumerate(rel_uncertainties):
        ax.text(
            j,
            0,
            f"{ru:.2%}",
            ha="center",
            va="center",
            color=_BG,
            fontsize=8,
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=_FG)
    cbar.set_label("Relative uncertainty", color=_FG)
    ax.set_title("Epistemic uncertainty heatmap", color=_FG)
    return ax


# ---------------------------------------------------------------------------
# Plotly variants
# ---------------------------------------------------------------------------


def plotly_knowledge_bar(
    knowledges: Sequence[Union[Knowledge, float, int]],
    labels: Sequence[str],
) -> "go.Figure":
    """Plotly bar chart with error bars.  Requires plotly.

    Parameters
    ----------
    knowledges : sequence of Knowledge or float
    labels : sequence of str

    Returns
    -------
    go.Figure
    """
    _require_plotly()
    if len(knowledges) != len(labels):
        raise ValueError("knowledges and labels must have the same length")

    values, epsilons = _extract_values_epsilons(knowledges)
    yerr = [2.0 * e for e in epsilons]

    fig = go.Figure(
        go.Bar(
            x=list(labels),
            y=values,
            error_y={"type": "data", "array": yerr, "visible": True, "color": _FG},
            marker_color=_GREEN,
            opacity=0.85,
        )
    )
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font={"color": _FG},
        title={"text": "Knowledge bar chart", "font": {"color": _FG}},
        xaxis={"gridcolor": "#313244"},
        yaxis={"gridcolor": "#313244", "title": "Value ± 2σ"},
    )
    return fig


def plotly_concentration_time(
    times: Sequence[float],
    concentrations: Sequence[Union[Knowledge, float, int]],
) -> "go.Figure":
    """Plotly concentration–time plot with ±2σ ribbon.  Requires plotly.

    Parameters
    ----------
    times : sequence of float
    concentrations : sequence of Knowledge or float

    Returns
    -------
    go.Figure
    """
    _require_plotly()

    if hasattr(concentrations, "values") and hasattr(concentrations, "epsilons"):
        values = list(concentrations.values)
        epsilons = list(concentrations.epsilons)
    else:
        values, epsilons = _extract_values_epsilons(concentrations)

    t = list(times)
    upper = [v + 2.0 * e for v, e in zip(values, epsilons)]
    lower = [v - 2.0 * e for v, e in zip(values, epsilons)]

    fig = go.Figure()
    # Shaded band: upper bound
    fig.add_trace(
        go.Scatter(
            x=t,
            y=upper,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="upper",
        )
    )
    # Shaded band: lower bound (fill to previous trace)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor=_GREEN.replace("#", "rgba(").rstrip(")") + ",0.2)",
            line={"width": 0},
            showlegend=True,
            name="±2σ",
        )
    )
    # Central line
    fig.add_trace(
        go.Scatter(
            x=t,
            y=values,
            mode="lines",
            line={"color": _GREEN, "width": 2},
            name="C(t)",
        )
    )
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font={"color": _FG},
        title={"text": "Concentration–time", "font": {"color": _FG}},
        xaxis={"gridcolor": "#313244", "title": "Time"},
        yaxis={"gridcolor": "#313244", "title": "Concentration"},
    )
    return fig
