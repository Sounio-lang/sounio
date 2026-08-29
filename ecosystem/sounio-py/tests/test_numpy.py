"""Tests for UncertainArray (NumPy integration).

Skipped automatically when numpy is not installed.
"""

import math
import sys
import os


try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

import pytest

if not _HAS_NUMPY:
    pytest.skip("numpy not installed", allow_module_level=True)

from sounio.integrations.numpy_ext import UncertainArray
from sounio.knowledge import Knowledge


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_basic():
    ua = UncertainArray([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], "test")
    assert len(ua) == 3
    np.testing.assert_array_equal(ua.values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ua.epsilons, [0.1, 0.2, 0.3])
    assert ua.provenance == "test"


def test_default_epsilons_are_zero():
    ua = UncertainArray([1.0, 2.0])
    np.testing.assert_array_equal(ua.epsilons, [0.0, 0.0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        UncertainArray([1.0, 2.0], [0.1])


def test_empty_array():
    ua = UncertainArray([], [])
    assert len(ua) == 0


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_getitem_returns_knowledge():
    ua = UncertainArray([5.0, 6.0], [0.5, 0.6], "arr")
    k = ua[0]
    assert isinstance(k, Knowledge)
    assert k.value == 5.0
    assert k.epsilon == 0.5
    assert k.provenance == "arr"


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def test_add_two_uncertain_arrays():
    a = UncertainArray([1.0, 2.0], [0.1, 0.2], "a")
    b = UncertainArray([3.0, 4.0], [0.3, 0.4], "b")
    c = a + b
    np.testing.assert_allclose(c.values, [4.0, 6.0])
    np.testing.assert_allclose(c.epsilons, [math.sqrt(0.01 + 0.09), math.sqrt(0.04 + 0.16)])


def test_add_scalar():
    a = UncertainArray([1.0, 2.0], [0.1, 0.2], "a")
    c = a + 10.0
    np.testing.assert_allclose(c.values, [11.0, 12.0])
    np.testing.assert_allclose(c.epsilons, [0.1, 0.2])


def test_radd_scalar():
    a = UncertainArray([1.0, 2.0], [0.1, 0.2], "a")
    c = 10.0 + a
    np.testing.assert_allclose(c.values, [11.0, 12.0])


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def test_sub_two_arrays():
    a = UncertainArray([5.0], [0.5], "a")
    b = UncertainArray([2.0], [0.3], "b")
    c = a - b
    assert math.isclose(c.values[0], 3.0)
    assert math.isclose(c.epsilons[0], math.sqrt(0.25 + 0.09))


def test_rsub_scalar():
    a = UncertainArray([3.0], [0.3], "a")
    c = 10.0 - a
    assert math.isclose(c.values[0], 7.0)
    assert math.isclose(c.epsilons[0], 0.3)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def test_mul_two_arrays():
    a = UncertainArray([4.0], [0.4], "a")  # 10% relative
    b = UncertainArray([2.0], [0.2], "b")  # 10% relative
    c = a * b
    assert math.isclose(c.values[0], 8.0)
    expected_eps = 8.0 * math.sqrt(0.01 + 0.01)
    assert math.isclose(c.epsilons[0], expected_eps)


def test_mul_scalar():
    a = UncertainArray([3.0, 4.0], [0.3, 0.4], "a")
    c = a * 2.0
    np.testing.assert_allclose(c.values, [6.0, 8.0])
    np.testing.assert_allclose(c.epsilons, [0.6, 0.8])


def test_mul_negative_scalar():
    a = UncertainArray([3.0], [0.3], "a")
    c = a * (-2.0)
    assert math.isclose(c.values[0], -6.0)
    assert math.isclose(c.epsilons[0], 0.6)


def test_mul_zero_elements():
    a = UncertainArray([0.0], [1.0], "a")
    b = UncertainArray([5.0], [0.5], "b")
    c = a * b
    assert c.values[0] == 0.0
    assert c.epsilons[0] == 0.0  # zero guard


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


def test_div_two_arrays():
    a = UncertainArray([10.0], [1.0], "a")  # 10%
    b = UncertainArray([2.0], [0.2], "b")   # 10%
    c = a / b
    assert math.isclose(c.values[0], 5.0)
    expected_eps = 5.0 * math.sqrt(0.01 + 0.01)
    assert math.isclose(c.epsilons[0], expected_eps)


def test_div_scalar():
    a = UncertainArray([6.0, 9.0], [0.6, 0.9], "a")
    c = a / 3.0
    np.testing.assert_allclose(c.values, [2.0, 3.0])
    np.testing.assert_allclose(c.epsilons, [0.2, 0.3])


def test_div_scalar_zero_raises():
    a = UncertainArray([1.0], [0.1], "a")
    with pytest.raises(ZeroDivisionError):
        _ = a / 0.0


def test_neg():
    a = UncertainArray([1.0, -2.0], [0.1, 0.2], "a")
    c = -a
    np.testing.assert_allclose(c.values, [-1.0, 2.0])
    np.testing.assert_allclose(c.epsilons, [0.1, 0.2])


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------


def test_mean():
    ua = UncertainArray([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "data")
    m = ua.mean()
    assert isinstance(m, Knowledge)
    assert math.isclose(m.value, 2.0)
    # ε_combined = sqrt(0.01*3) / 3 = sqrt(0.03)/3
    expected_eps = math.sqrt(0.03) / 3
    assert math.isclose(m.epsilon, expected_eps)
    assert "mean" in m.provenance


def test_mean_empty():
    ua = UncertainArray([], [])
    m = ua.mean()
    assert math.isnan(m.value)


def test_sum():
    ua = UncertainArray([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], "s")
    s = ua.sum()
    assert math.isclose(s.value, 6.0)
    expected_eps = math.sqrt(0.01 + 0.04 + 0.09)
    assert math.isclose(s.epsilon, expected_eps)


def test_min_max():
    ua = UncertainArray([3.0, 1.0, 2.0], [0.3, 0.1, 0.2], "m")
    mn = ua.min()
    mx = ua.max()
    assert math.isclose(mn.value, 1.0)
    assert math.isclose(mn.epsilon, 0.1)
    assert math.isclose(mx.value, 3.0)
    assert math.isclose(mx.epsilon, 0.3)


def test_std():
    ua = UncertainArray([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "data")
    s = ua.std()
    assert isinstance(s, Knowledge)
    assert math.isclose(s.value, np.std([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Conversion to/from Knowledge
# ---------------------------------------------------------------------------


def test_to_knowledge_list():
    ua = UncertainArray([1.0, 2.0], [0.1, 0.2], "arr")
    ks = ua.to_knowledge_list()
    assert len(ks) == 2
    assert all(isinstance(k, Knowledge) for k in ks)
    assert ks[0].value == 1.0
    assert ks[1].epsilon == 0.2


def test_from_knowledge_list():
    ks = [Knowledge(1.0, 0.1, "src"), Knowledge(2.0, 0.2, "src")]
    ua = UncertainArray.from_knowledge_list(ks)
    np.testing.assert_allclose(ua.values, [1.0, 2.0])
    np.testing.assert_allclose(ua.epsilons, [0.1, 0.2])
    assert ua.provenance == "src"


def test_roundtrip_knowledge_list():
    ks = [Knowledge(float(i), float(i) * 0.01, "x") for i in range(1, 6)]
    ua = UncertainArray.from_knowledge_list(ks)
    ks2 = ua.to_knowledge_list()
    for a, b in zip(ks, ks2):
        assert math.isclose(a.value, b.value)
        assert math.isclose(a.epsilon, b.epsilon)


def test_from_empty_list():
    ua = UncertainArray.from_knowledge_list([])
    assert len(ua) == 0


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr():
    ua = UncertainArray([1.0, 2.0], [0.1, 0.2], "test")
    r = repr(ua)
    assert "UncertainArray" in r
    assert "n=2" in r
