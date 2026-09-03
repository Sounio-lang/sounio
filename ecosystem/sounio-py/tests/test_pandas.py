"""Tests for EpistemicDataFrame (pandas integration).

Skipped automatically when pandas/numpy is not installed.
"""

import math
import sys
import os


try:
    import pandas as pd
    import numpy as np
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

import pytest

if not _HAS_PANDAS:
    pytest.skip("pandas not installed", allow_module_level=True)

from sounio.integrations.pandas_ext import EpistemicDataFrame
from sounio.knowledge import Knowledge


# ---------------------------------------------------------------------------
# from_measurements
# ---------------------------------------------------------------------------


def test_from_measurements_basic():
    edf = EpistemicDataFrame.from_measurements({
        "dose": [(100.0, 2.0), (200.0, 3.0)],
    })
    assert len(edf) == 2
    assert "dose" in edf.df.columns
    assert "dose_epsilon" in edf.df.columns
    assert "dose_prov" in edf.df.columns


def test_from_measurements_with_provenance():
    edf = EpistemicDataFrame.from_measurements({
        "weight": [(70.0, 0.5, "scale-A"), (80.0, 0.5, "scale-B")],
    })
    provs = list(edf.df["weight_prov"])
    assert provs == ["scale-A", "scale-B"]


def test_from_measurements_default_provenance():
    edf = EpistemicDataFrame.from_measurements({
        "temp": [(36.5, 0.1), (37.0, 0.1)],
    })
    # Default provenance should be column name
    assert all(p == "temp" for p in edf.df["temp_prov"])


def test_from_measurements_multiple_cols():
    edf = EpistemicDataFrame.from_measurements({
        "dose": [(100.0, 2.0), (200.0, 3.0)],
        "weight": [(70.0, 0.5), (85.0, 0.5)],
    })
    assert "dose" in edf.measured_columns
    assert "weight" in edf.measured_columns
    assert len(edf.measured_columns) == 2


# ---------------------------------------------------------------------------
# from_knowledge_columns
# ---------------------------------------------------------------------------


def test_from_knowledge_columns():
    ks = [Knowledge(1.0, 0.1, "lab"), Knowledge(2.0, 0.2, "lab")]
    edf = EpistemicDataFrame.from_knowledge_columns({"x": ks})
    assert "x" in edf.measured_columns
    assert edf.df["x"].tolist() == [1.0, 2.0]
    assert edf.df["x_epsilon"].tolist() == [0.1, 0.2]


# ---------------------------------------------------------------------------
# get_knowledge_column
# ---------------------------------------------------------------------------


def test_get_knowledge_column():
    edf = EpistemicDataFrame.from_measurements({
        "dose": [(100.0, 2.0, "syringe"), (200.0, 3.0, "syringe")],
    })
    ks = edf.get_knowledge_column("dose")
    assert len(ks) == 2
    assert all(isinstance(k, Knowledge) for k in ks)
    assert ks[0].value == 100.0
    assert ks[0].epsilon == 2.0
    assert ks[0].provenance == "syringe"


def test_get_knowledge_column_missing_raises():
    edf = EpistemicDataFrame.from_measurements({"x": [(1.0, 0.1)]})
    with pytest.raises(KeyError):
        edf.get_knowledge_column("nonexistent")


def test_get_knowledge_column_no_epsilon():
    # Manually built DataFrame without epsilon column
    df = pd.DataFrame({"y": [1.0, 2.0]})
    edf = EpistemicDataFrame(df)
    # "y" has no "y_epsilon" → measured_columns should not include it
    assert "y" not in edf.measured_columns


# ---------------------------------------------------------------------------
# set_knowledge_column
# ---------------------------------------------------------------------------


def test_set_knowledge_column():
    edf = EpistemicDataFrame.from_measurements({"x": [(1.0, 0.1)]})
    new_ks = [Knowledge(99.0, 0.5, "new")]
    edf.set_knowledge_column("x", new_ks)
    assert edf.df["x"].iloc[0] == 99.0
    assert edf.df["x_epsilon"].iloc[0] == 0.5
    assert edf.df["x_prov"].iloc[0] == "new"


# ---------------------------------------------------------------------------
# describe_epistemic
# ---------------------------------------------------------------------------


def test_describe_epistemic_columns():
    edf = EpistemicDataFrame.from_measurements({
        "dose": [(100.0, 2.0), (200.0, 3.0), (150.0, 2.5)],
    })
    stats = edf.describe_epistemic()
    assert "dose" in stats.index
    assert "mean" in stats.columns
    assert "std" in stats.columns
    assert "mean_epsilon" in stats.columns
    assert "max_epsilon" in stats.columns
    assert "rel_uncertainty" in stats.columns


def test_describe_epistemic_values():
    edf = EpistemicDataFrame.from_measurements({
        "x": [(10.0, 1.0), (20.0, 2.0), (30.0, 3.0)],
    })
    stats = edf.describe_epistemic()
    assert math.isclose(stats.loc["x", "mean"], 20.0)
    assert math.isclose(stats.loc["x", "mean_epsilon"], 2.0)
    assert math.isclose(stats.loc["x", "max_epsilon"], 3.0)


def test_describe_epistemic_empty_df():
    edf = EpistemicDataFrame(pd.DataFrame())
    stats = edf.describe_epistemic()
    assert stats.empty


def test_describe_epistemic_no_epsilon_col():
    df = pd.DataFrame({"y": [1.0, 2.0]})
    edf = EpistemicDataFrame(df)
    stats = edf.describe_epistemic()
    assert stats.empty


# ---------------------------------------------------------------------------
# epistemic_summary
# ---------------------------------------------------------------------------


def test_epistemic_summary():
    edf = EpistemicDataFrame.from_measurements({
        "dose": [(100.0, 2.0, "d"), (200.0, 4.0, "d"), (300.0, 6.0, "d")],
    })
    s = edf.epistemic_summary("dose")
    assert isinstance(s, Knowledge)
    assert math.isclose(s.value, 200.0)
    expected_eps = math.sqrt(4 + 16 + 36) / 3
    assert math.isclose(s.epsilon, expected_eps)
    assert "dose" in s.provenance


def test_epistemic_summary_single_row():
    edf = EpistemicDataFrame.from_measurements({
        "x": [(7.0, 0.7, "lab")],
    })
    s = edf.epistemic_summary("x")
    assert math.isclose(s.value, 7.0)
    assert math.isclose(s.epsilon, 0.7)


# ---------------------------------------------------------------------------
# measured_columns
# ---------------------------------------------------------------------------


def test_measured_columns_excludes_meta():
    edf = EpistemicDataFrame.from_measurements({
        "a": [(1.0, 0.1)],
        "b": [(2.0, 0.2)],
    })
    cols = edf.measured_columns
    assert "a_epsilon" not in cols
    assert "a_prov" not in cols
    assert set(cols) == {"a", "b"}


# ---------------------------------------------------------------------------
# head / len / repr
# ---------------------------------------------------------------------------


def test_len():
    edf = EpistemicDataFrame.from_measurements({"x": [(float(i), 0.1) for i in range(10)]})
    assert len(edf) == 10


def test_head():
    edf = EpistemicDataFrame.from_measurements({"x": [(float(i), 0.1) for i in range(10)]})
    h = edf.head(3)
    assert isinstance(h, pd.DataFrame)
    assert len(h) == 3


def test_repr():
    edf = EpistemicDataFrame.from_measurements({"x": [(1.0, 0.1)]})
    r = repr(edf)
    assert "EpistemicDataFrame" in r
    assert "x" in r


# ---------------------------------------------------------------------------
# Round-trip: Knowledge → EDF → Knowledge
# ---------------------------------------------------------------------------


def test_roundtrip_knowledge():
    original = [
        Knowledge(1.0, 0.1, "src"),
        Knowledge(2.0, 0.2, "src"),
        Knowledge(3.0, 0.3, "src"),
    ]
    edf = EpistemicDataFrame.from_knowledge_columns({"measure": original})
    recovered = edf.get_knowledge_column("measure")
    for a, b in zip(original, recovered):
        assert math.isclose(a.value, b.value)
        assert math.isclose(a.epsilon, b.epsilon)
        assert a.provenance == b.provenance


# ---------------------------------------------------------------------------
# Rel uncertainty edge case: zero denominator
# ---------------------------------------------------------------------------


def test_describe_epistemic_zero_values():
    edf = EpistemicDataFrame.from_measurements({
        "x": [(0.0, 1.0), (1.0, 0.1)],
    })
    # Should not raise; zero values → NaN for rel_uncertainty (handled via replace)
    stats = edf.describe_epistemic()
    assert "x" in stats.index
