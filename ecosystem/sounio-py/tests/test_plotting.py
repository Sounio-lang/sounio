"""Tests for sounio.integrations.plotting.

Matplotlib is mocked at import time so tests run headlessly without a display
server or matplotlib installed.  We verify that the correct data is passed to
bar / fill_between / errorbar / imshow / scatter calls, and exercise edge
cases (empty data, single point, zero epsilon).
"""

from __future__ import annotations

import math
import os
import sys
import types
from unittest.mock import MagicMock, patch, call
import pytest

# Ensure the pure-Python package is importable regardless of how pytest is
# invoked (e.g. PYTHONPATH=python or plain python3 -m pytest).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_PKG = os.path.normpath(os.path.join(_HERE, "..", "python"))


# ---------------------------------------------------------------------------
# Inject a fake matplotlib into sys.modules BEFORE plotting is imported so
# that _HAS_MATPLOTLIB becomes True and the real import guard is satisfied.
# ---------------------------------------------------------------------------

def _build_mock_matplotlib():
    """Return a minimal fake matplotlib package tree."""
    mpl = types.ModuleType("matplotlib")
    mpl_pyplot = types.ModuleType("matplotlib.pyplot")
    mpl_colors = types.ModuleType("matplotlib.colors")

    # pyplot.subplots — default returns (MagicMock, MagicMock); tests override
    mpl_pyplot.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))

    # colors.LinearSegmentedColormap.from_list
    cmap_mock = MagicMock()
    lscm = MagicMock()
    lscm.from_list = MagicMock(return_value=cmap_mock)
    mpl_colors.LinearSegmentedColormap = lscm

    mpl.pyplot = mpl_pyplot
    mpl.colors = mpl_colors

    return mpl, mpl_pyplot, mpl_colors


_fake_mpl, _fake_pyplot, _fake_mpl_colors = _build_mock_matplotlib()

# Instead of injecting into sys.modules, we'll patch the imported modules in sounio.integrations.plotting later.
# We still ensure plotly is absent
_plotly_real = sys.modules.get("plotly")
_plotly_go_real = sys.modules.get("plotly.graph_objects")

# Now safe to import plotting
from sounio.knowledge import Knowledge
from sounio.types import PKParameters
import sounio.integrations.plotting

# Patch the modules inside sounio.integrations.plotting
sounio.integrations.plotting.matplotlib = _fake_mpl
sounio.integrations.plotting.plt = _fake_pyplot
sounio.integrations.plotting._HAS_MATPLOTLIB = True


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ax():
    """Return a MagicMock that quacks like a matplotlib Axes."""
    ax = MagicMock()
    fig = MagicMock()
    fig.colorbar = MagicMock(return_value=MagicMock())
    ax.get_figure.return_value = fig
    ax.spines = {
        "left": MagicMock(),
        "right": MagicMock(),
        "top": MagicMock(),
        "bottom": MagicMock(),
    }
    ax.xaxis = MagicMock()
    ax.yaxis = MagicMock()
    ax.title = MagicMock()
    ax.tick_params = MagicMock()
    return ax


def _make_fig_ax():
    fig = MagicMock()
    fig.colorbar = MagicMock(return_value=MagicMock())
    ax = _make_ax()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_knowledge_bar
# ---------------------------------------------------------------------------

class TestPlotKnowledgeBar:
    def _import(self):
        # Re-import fresh so mock is active
        import importlib
        import sounio.integrations.plotting as m
        importlib.reload(m)
        return m.plot_knowledge_bar

    def test_basic_bar(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        knowledges = [Knowledge(10.0, 1.0), Knowledge(20.0, 2.0)]
        labels = ["A", "B"]
        ax = _make_ax()

        result = plot_knowledge_bar(knowledges, labels, ax=ax)

        assert result is ax
        ax.bar.assert_called_once()
        args, kwargs = ax.bar.call_args
        assert args[0] == [0, 1]
        assert args[1] == [10.0, 20.0]
        assert kwargs["yerr"] == [2.0, 4.0]

    def test_zero_epsilon(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        ax = _make_ax()
        plot_knowledge_bar([Knowledge(5.0, 0.0), Knowledge(15.0, 0.0)], ["X", "Y"], ax=ax)
        _, kwargs = ax.bar.call_args
        assert kwargs["yerr"] == [0.0, 0.0]

    def test_creates_axes_when_none(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        fig, ax = _make_fig_ax()
        _fake_pyplot.subplots = MagicMock(return_value=(fig, ax))
        result = plot_knowledge_bar([Knowledge(1.0, 0.1)], ["a"])
        assert result is ax

    def test_mismatched_lengths_raises(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        with pytest.raises(ValueError, match="same length"):
            plot_knowledge_bar([Knowledge(1.0)], ["a", "b"], ax=_make_ax())

    def test_empty_data(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        ax = _make_ax()
        result = plot_knowledge_bar([], [], ax=ax)
        assert result is ax
        args, kwargs = ax.bar.call_args
        assert args[0] == []
        assert args[1] == []

    def test_single_point(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        ax = _make_ax()
        plot_knowledge_bar([Knowledge(42.0, 3.0)], ["only"], ax=ax)
        _, kwargs = ax.bar.call_args
        assert kwargs["yerr"] == [6.0]

    def test_plain_float_inputs(self):
        from sounio.integrations.plotting import plot_knowledge_bar

        ax = _make_ax()
        plot_knowledge_bar([1.0, 2.0, 3.0], ["a", "b", "c"], ax=ax)
        _, kwargs = ax.bar.call_args
        assert kwargs["yerr"] == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# plot_concentration_time
# ---------------------------------------------------------------------------

class TestPlotConcentrationTime:
    def test_basic(self):
        from sounio.integrations.plotting import plot_concentration_time

        times = [0.0, 1.0, 2.0]
        concs = [Knowledge(10.0, 1.0), Knowledge(8.0, 0.5), Knowledge(6.0, 0.3)]
        ax = _make_ax()

        result = plot_concentration_time(times, concs, ax=ax)

        assert result is ax
        ax.plot.assert_called_once()
        ax.fill_between.assert_called_once()
        fb_args = ax.fill_between.call_args[0]
        assert list(fb_args[0]) == times
        # lower = value - 2*eps
        assert list(fb_args[1]) == pytest.approx([10.0 - 2.0, 8.0 - 1.0, 6.0 - 0.6])
        # upper = value + 2*eps
        assert list(fb_args[2]) == pytest.approx([10.0 + 2.0, 8.0 + 1.0, 6.0 + 0.6])

    def test_zero_uncertainty(self):
        from sounio.integrations.plotting import plot_concentration_time

        ax = _make_ax()
        plot_concentration_time([0.0, 1.0], [Knowledge(5.0, 0.0), Knowledge(3.0, 0.0)], ax=ax)
        fb_args = ax.fill_between.call_args[0]
        assert list(fb_args[1]) == list(fb_args[2])

    def test_single_point(self):
        from sounio.integrations.plotting import plot_concentration_time

        ax = _make_ax()
        result = plot_concentration_time([0.0], [Knowledge(7.0, 0.5)], ax=ax)
        assert result is ax

    def test_uncertain_array_duck_type(self):
        from sounio.integrations.plotting import plot_concentration_time

        ua = MagicMock(spec=[])  # no __iter__
        ua.values = [1.0, 2.0]
        ua.epsilons = [0.1, 0.2]
        ax = _make_ax()
        plot_concentration_time([0.0, 1.0], ua, ax=ax)
        fb_args = ax.fill_between.call_args[0]
        assert list(fb_args[1]) == pytest.approx([1.0 - 0.2, 2.0 - 0.4])
        assert list(fb_args[2]) == pytest.approx([1.0 + 0.2, 2.0 + 0.4])

    def test_float_inputs(self):
        from sounio.integrations.plotting import plot_concentration_time

        ax = _make_ax()
        plot_concentration_time([0.0, 1.0], [5.0, 3.0], ax=ax)
        fb_args = ax.fill_between.call_args[0]
        assert list(fb_args[1]) == list(fb_args[2])


# ---------------------------------------------------------------------------
# plot_funnel
# ---------------------------------------------------------------------------

class TestPlotFunnel:
    def test_basic(self):
        from sounio.integrations.plotting import plot_funnel

        results = [
            {"effect": Knowledge(0.5, 0.1), "se": 0.2, "label": "Trial A"},
            {"effect": 0.3, "se": 0.15},
        ]
        ax = _make_ax()
        result = plot_funnel(results, ax=ax)
        assert result is ax
        assert ax.scatter.call_count == 2

    def test_empty(self):
        from sounio.integrations.plotting import plot_funnel

        ax = _make_ax()
        result = plot_funnel([], ax=ax)
        assert result is ax
        ax.scatter.assert_not_called()

    def test_single_trial(self):
        from sounio.integrations.plotting import plot_funnel

        ax = _make_ax()
        plot_funnel([{"effect": Knowledge(1.0, 0.05), "se": 0.5}], ax=ax)
        assert ax.scatter.call_count == 1

    def test_precision_computed_correctly(self):
        from sounio.integrations.plotting import plot_funnel

        ax = _make_ax()
        plot_funnel([{"effect": 1.0, "se": 0.25}], ax=ax)
        scatter_args = ax.scatter.call_args[0]
        assert scatter_args[1] == [4.0]

    def test_label_annotates(self):
        from sounio.integrations.plotting import plot_funnel

        ax = _make_ax()
        plot_funnel([{"effect": 0.5, "se": 0.2, "label": "X"}], ax=ax)
        ax.annotate.assert_called_once()


# ---------------------------------------------------------------------------
# plot_pk_profile
# ---------------------------------------------------------------------------

class TestPlotPKProfile:
    def _make_pk(self):
        return PKParameters(
            clearance=Knowledge(5.0, 0.5),
            volume_dist=Knowledge(50.0, 5.0),
            half_life=Knowledge(6.93, 0.7),
            bioavailability=Knowledge(0.8, 0.05),
        )

    def test_returns_ax(self):
        from sounio.integrations.plotting import plot_pk_profile

        ax = _make_ax()
        result = plot_pk_profile(self._make_pk(), dose=100.0, ax=ax)
        assert result is ax

    def test_plot_called_with_n_points(self):
        from sounio.integrations.plotting import plot_pk_profile

        ax = _make_ax()
        plot_pk_profile(self._make_pk(), dose=100.0, n_points=50, ax=ax)
        plot_args = ax.plot.call_args[0]
        assert len(plot_args[0]) == 50
        assert len(plot_args[1]) == 50

    def test_concentration_decreases(self):
        from sounio.integrations.plotting import plot_pk_profile

        ax = _make_ax()
        plot_pk_profile(self._make_pk(), dose=100.0, t_max=24.0, n_points=10, ax=ax)
        c_vals = ax.plot.call_args[0][1]
        assert c_vals[0] >= c_vals[-1]

    def test_dose_as_knowledge(self):
        from sounio.integrations.plotting import plot_pk_profile

        ax = _make_ax()
        result = plot_pk_profile(self._make_pk(), dose=Knowledge(200.0, 10.0), ax=ax)
        assert result is ax

    def test_zero_vd_no_crash(self):
        from sounio.integrations.plotting import plot_pk_profile

        pk = PKParameters(
            clearance=Knowledge(5.0, 0.0),
            volume_dist=Knowledge(0.0, 0.0),
            half_life=Knowledge(0.0, 0.0),
            bioavailability=Knowledge(1.0, 0.0),
        )
        ax = _make_ax()
        result = plot_pk_profile(pk, dose=100.0, ax=ax)
        assert result is ax
        c_vals = ax.plot.call_args[0][1]
        assert all(v == 0.0 for v in c_vals)

    def test_fill_between_called(self):
        from sounio.integrations.plotting import plot_pk_profile

        ax = _make_ax()
        plot_pk_profile(self._make_pk(), dose=100.0, ax=ax)
        ax.fill_between.assert_called_once()

    def test_creates_axes_when_none(self):
        from sounio.integrations.plotting import plot_pk_profile

        fig, ax = _make_fig_ax()
        _fake_pyplot.subplots = MagicMock(return_value=(fig, ax))
        result = plot_pk_profile(self._make_pk(), dose=100.0)
        assert result is ax

    def test_c0_formula(self):
        """At t=0: C(0) = F*D/Vd (no uncertainty test, just magnitude)."""
        from sounio.integrations.plotting import plot_pk_profile

        pk = PKParameters(
            clearance=Knowledge(5.0, 0.0),
            volume_dist=Knowledge(50.0, 0.0),
            half_life=Knowledge(6.93, 0.0),
            bioavailability=Knowledge(0.8, 0.0),
        )
        ax = _make_ax()
        plot_pk_profile(pk, dose=100.0, n_points=2, t_max=10.0, ax=ax)
        c_vals = ax.plot.call_args[0][1]
        expected_c0 = 0.8 * 100.0 / 50.0
        assert abs(c_vals[0] - expected_c0) < 1e-10


# ---------------------------------------------------------------------------
# plot_knowledge_scatter
# ---------------------------------------------------------------------------

class TestPlotKnowledgeScatter:
    def test_basic(self):
        from sounio.integrations.plotting import plot_knowledge_scatter

        x = [Knowledge(1.0, 0.1), Knowledge(2.0, 0.2)]
        y = [Knowledge(3.0, 0.3), Knowledge(4.0, 0.4)]
        ax = _make_ax()
        result = plot_knowledge_scatter(x, y, ax=ax)
        assert result is ax
        ax.errorbar.assert_called_once()
        _, kwargs = ax.errorbar.call_args
        assert kwargs["xerr"] == pytest.approx([0.2, 0.4])
        assert kwargs["yerr"] == pytest.approx([0.6, 0.8])

    def test_with_labels(self):
        from sounio.integrations.plotting import plot_knowledge_scatter

        ax = _make_ax()
        plot_knowledge_scatter([Knowledge(1.0, 0.0)], [Knowledge(2.0, 0.0)], labels=["P1"], ax=ax)
        ax.annotate.assert_called_once()

    def test_mismatched_raises(self):
        from sounio.integrations.plotting import plot_knowledge_scatter

        with pytest.raises(ValueError, match="same length"):
            plot_knowledge_scatter(
                [Knowledge(1.0)],
                [Knowledge(2.0), Knowledge(3.0)],
                ax=_make_ax(),
            )

    def test_empty(self):
        from sounio.integrations.plotting import plot_knowledge_scatter

        ax = _make_ax()
        result = plot_knowledge_scatter([], [], ax=ax)
        assert result is ax

    def test_zero_epsilon(self):
        from sounio.integrations.plotting import plot_knowledge_scatter

        ax = _make_ax()
        plot_knowledge_scatter([Knowledge(1.0, 0.0)], [Knowledge(2.0, 0.0)], ax=ax)
        _, kwargs = ax.errorbar.call_args
        assert kwargs["xerr"] == [0.0]
        assert kwargs["yerr"] == [0.0]

    def test_no_labels_no_annotate(self):
        from sounio.integrations.plotting import plot_knowledge_scatter

        ax = _make_ax()
        plot_knowledge_scatter([Knowledge(1.0)], [Knowledge(2.0)], ax=ax)
        ax.annotate.assert_not_called()


# ---------------------------------------------------------------------------
# plot_epistemic_heatmap
# ---------------------------------------------------------------------------

class TestPlotEpistemicHeatmap:
    def _call(self, data_dict):
        from sounio.integrations.plotting import plot_epistemic_heatmap

        fig, ax = _make_fig_ax()
        _fake_pyplot.subplots = MagicMock(return_value=(fig, ax))
        result = plot_epistemic_heatmap(data_dict)
        return result, ax, fig

    def test_basic(self):
        data = {
            "clearance": Knowledge(5.0, 0.5),
            "volume": Knowledge(50.0, 0.0),
            "bioavailability": Knowledge(0.8, 0.16),
        }
        result, ax, _ = self._call(data)
        assert result is ax
        ax.imshow.assert_called_once()

    def test_relative_uncertainty_values(self):
        data = {
            "a": Knowledge(10.0, 1.0),   # rel_u = 0.1
            "b": Knowledge(10.0, 0.0),   # rel_u = 0.0
        }
        _, ax, _ = self._call(data)
        imshow_args = ax.imshow.call_args[0]
        data_2d = imshow_args[0]
        assert data_2d == [[0.1, 0.0]]

    def test_float_values_have_zero_uncertainty(self):
        _, ax, _ = self._call({"x": 5.0, "y": 10.0})
        imshow_args = ax.imshow.call_args[0]
        assert imshow_args[0] == [[0.0, 0.0]]

    def test_empty_dict(self):
        result, ax, _ = self._call({})
        assert result is ax

    def test_single_entry(self):
        _, ax, _ = self._call({"dose": Knowledge(500.0, 25.0)})
        imshow_args = ax.imshow.call_args[0]
        assert abs(imshow_args[0][0][0] - 0.05) < 1e-10

    def test_provided_ax_used(self):
        from sounio.integrations.plotting import plot_epistemic_heatmap

        data = {"k": Knowledge(1.0, 0.1)}
        ax = _make_ax()
        result = plot_epistemic_heatmap(data, ax=ax)
        assert result is ax
        ax.imshow.assert_called_once()


# ---------------------------------------------------------------------------
# Plotly variants
# ---------------------------------------------------------------------------

_plotly_available = pytest.mark.skipif(
    not (sys.modules.get("plotly") or __import__("importlib").util.find_spec("plotly")),
    reason="plotly not installed",
)


@_plotly_available
class TestPlotlyKnowledgeBar:
    def test_returns_figure(self):
        from sounio.integrations.plotting import plotly_knowledge_bar
        import plotly.graph_objects as go

        fig = plotly_knowledge_bar(
            [Knowledge(10.0, 1.0), Knowledge(20.0, 2.0)],
            ["A", "B"],
        )
        assert isinstance(fig, go.Figure)

    def test_error_values(self):
        from sounio.integrations.plotting import plotly_knowledge_bar

        fig = plotly_knowledge_bar([Knowledge(5.0, 0.5)], ["X"])
        bar = fig.data[0]
        assert bar.error_y["array"] == [1.0]  # 2 * 0.5

    def test_mismatched_raises(self):
        from sounio.integrations.plotting import plotly_knowledge_bar

        with pytest.raises(ValueError):
            plotly_knowledge_bar([Knowledge(1.0)], ["a", "b"])

    def test_empty(self):
        from sounio.integrations.plotting import plotly_knowledge_bar
        import plotly.graph_objects as go

        fig = plotly_knowledge_bar([], [])
        assert isinstance(fig, go.Figure)


@_plotly_available
class TestPlotlyConcentrationTime:
    def test_returns_figure(self):
        from sounio.integrations.plotting import plotly_concentration_time
        import plotly.graph_objects as go

        fig = plotly_concentration_time(
            [0.0, 1.0, 2.0],
            [Knowledge(10.0, 1.0), Knowledge(8.0, 0.5), Knowledge(6.0, 0.3)],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_zero_uncertainty_band(self):
        from sounio.integrations.plotting import plotly_concentration_time

        fig = plotly_concentration_time(
            [0.0, 1.0],
            [Knowledge(5.0, 0.0), Knowledge(3.0, 0.0)],
        )
        upper_trace = fig.data[0]
        lower_trace = fig.data[1]
        assert list(upper_trace.y) == list(lower_trace.y)
