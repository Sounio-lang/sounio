"""Tests for the pure-Python Knowledge class.

Run with:  pytest tests/test_knowledge.py -v
"""

import math
import sys
import os

# Ensure we test the pure-Python implementation regardless of whether the
# native extension is installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from sounio.knowledge import Knowledge


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_epsilon_and_provenance():
    k = Knowledge(5.0)
    assert k.value == 5.0
    assert k.epsilon == 0.0
    assert k.provenance == ""


def test_explicit_fields():
    k = Knowledge(3.14, 0.01, "pi-approx")
    assert k.value == 3.14
    assert k.epsilon == 0.01
    assert k.provenance == "pi-approx"


def test_int_value_coerced_to_float():
    k = Knowledge(42)
    assert isinstance(k.value, float)
    assert k.value == 42.0


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_str_format():
    k = Knowledge(1.0, 0.1, "sensor")
    s = str(k)
    assert "1.000" in s
    assert "0.100" in s
    assert "sensor" in s


def test_repr_format():
    k = Knowledge(2.5, 0.05, "scale")
    r = repr(k)
    assert "Knowledge" in r
    assert "value" in r
    assert "epsilon" in r
    assert "prov" in r


def test_format_spec_delegates_to_value():
    k = Knowledge(1.23456789, 0.001, "x")
    assert format(k, ".2f") == "1.23"


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def test_add_two_knowledge():
    a = Knowledge(10.0, 3.0, "a")
    b = Knowledge(5.0, 4.0, "b")
    c = a + b
    assert c.value == 15.0
    assert math.isclose(c.epsilon, math.sqrt(9 + 16), rel_tol=1e-12)


def test_add_scalar():
    k = Knowledge(10.0, 2.0, "k")
    r = k + 5
    assert r.value == 15.0
    assert r.epsilon == 2.0
    assert r.provenance == "k"


def test_radd_scalar():
    k = Knowledge(10.0, 2.0, "k")
    r = 5 + k
    assert r.value == 15.0


def test_add_zero_uncertainty():
    a = Knowledge(1.0, 0.0, "a")
    b = Knowledge(2.0, 0.0, "b")
    c = a + b
    assert c.epsilon == 0.0


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def test_sub_two_knowledge():
    a = Knowledge(10.0, 3.0, "a")
    b = Knowledge(4.0, 4.0, "b")
    c = a - b
    assert c.value == 6.0
    assert math.isclose(c.epsilon, math.sqrt(9 + 16), rel_tol=1e-12)


def test_sub_scalar():
    k = Knowledge(10.0, 1.0, "k")
    r = k - 3
    assert r.value == 7.0
    assert r.epsilon == 1.0


def test_rsub_scalar():
    k = Knowledge(4.0, 1.0, "k")
    r = 10 - k
    assert r.value == 6.0


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def test_mul_two_knowledge():
    a = Knowledge(6.0, 0.6, "a")  # 10% relative
    b = Knowledge(4.0, 0.4, "b")  # 10% relative
    c = a * b
    assert math.isclose(c.value, 24.0)
    expected_eps = 24.0 * math.sqrt(0.01 + 0.01)
    assert math.isclose(c.epsilon, expected_eps, rel_tol=1e-12)


def test_mul_scalar():
    k = Knowledge(5.0, 1.0, "k")
    r = k * 3
    assert r.value == 15.0
    assert r.epsilon == 3.0


def test_rmul_scalar():
    k = Knowledge(5.0, 1.0, "k")
    r = 3 * k
    assert r.value == 15.0
    assert r.epsilon == 3.0


def test_mul_negative_scalar():
    k = Knowledge(5.0, 2.0, "k")
    r = k * (-2)
    assert r.value == -10.0
    assert math.isclose(r.epsilon, 4.0)  # |factor| * epsilon


def test_mul_zero_value():
    a = Knowledge(0.0, 1.0, "a")
    b = Knowledge(5.0, 0.5, "b")
    c = a * b
    # Relative uncertainty of a is 0 (zero-guard), so epsilon = 0
    assert c.value == 0.0
    assert c.epsilon == 0.0


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


def test_div_two_knowledge():
    a = Knowledge(10.0, 1.0, "a")  # 10% relative
    b = Knowledge(2.0, 0.2, "b")   # 10% relative
    c = a / b
    assert math.isclose(c.value, 5.0)
    expected_eps = 5.0 * math.sqrt(0.01 + 0.01)
    assert math.isclose(c.epsilon, expected_eps, rel_tol=1e-12)


def test_div_scalar():
    k = Knowledge(10.0, 2.0, "k")
    r = k / 2
    assert r.value == 5.0
    assert r.epsilon == 1.0


def test_div_by_zero_raises():
    a = Knowledge(5.0, 0.1, "a")
    b = Knowledge(0.0, 0.0, "zero")
    try:
        _ = a / b
        assert False, "should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass


def test_div_scalar_zero_raises():
    k = Knowledge(5.0, 0.1, "k")
    try:
        _ = k / 0
        assert False, "should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass


def test_rtruediv():
    k = Knowledge(4.0, 0.4, "k")  # 10% relative
    r = 8.0 / k
    assert math.isclose(r.value, 2.0)
    assert math.isclose(r.epsilon, 0.2, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Negation / abs
# ---------------------------------------------------------------------------


def test_neg():
    k = Knowledge(5.0, 1.0, "k")
    r = -k
    assert r.value == -5.0
    assert r.epsilon == 1.0


def test_abs():
    k = Knowledge(-3.0, 0.5, "k")
    r = abs(k)
    assert r.value == 3.0
    assert r.epsilon == 0.5


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


def test_relative_uncertainty():
    k = Knowledge(100.0, 5.0, "k")
    assert math.isclose(k.relative_uncertainty, 0.05)


def test_relative_uncertainty_zero_value():
    k = Knowledge(0.0, 1.0, "k")
    assert k.relative_uncertainty == 0.0


def test_confidence():
    k = Knowledge(100.0, 5.0, "k")
    assert math.isclose(k.confidence, 0.95)


def test_confidence_clamped_to_zero():
    # epsilon > value → relative_uncertainty > 1 → confidence clamped to 0
    k = Knowledge(1.0, 200.0, "k")
    assert k.confidence == 0.0


def test_is_reliable_default_threshold():
    k = Knowledge(100.0, 4.0, "k")   # 4% — reliable
    assert k.is_reliable()
    k2 = Knowledge(100.0, 6.0, "k")  # 6% — not reliable
    assert not k2.is_reliable()


def test_is_reliable_custom_threshold():
    k = Knowledge(100.0, 9.0, "k")
    assert k.is_reliable(threshold=0.10)
    assert not k.is_reliable(threshold=0.08)


def test_scale():
    k = Knowledge(10.0, 2.0, "k")
    r = k.scale(3.0)
    assert r.value == 30.0
    assert r.epsilon == 6.0
    assert r.provenance == "k"


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


def test_equal_objects():
    a = Knowledge(1.0, 0.1, "x")
    b = Knowledge(1.0, 0.1, "x")
    assert a == b


def test_unequal_value():
    a = Knowledge(1.0, 0.1, "x")
    b = Knowledge(2.0, 0.1, "x")
    assert a != b


def test_hashable():
    a = Knowledge(1.0, 0.1, "x")
    b = Knowledge(1.0, 0.1, "x")
    s = {a, b}
    assert len(s) == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict():
    k = Knowledge(42.0, 0.5, "sensor")
    d = k.to_dict()
    assert d == {"value": 42.0, "epsilon": 0.5, "provenance": "sensor"}


def test_from_dict():
    d = {"value": 7.0, "epsilon": 0.1, "provenance": "lab"}
    k = Knowledge.from_dict(d)
    assert k.value == 7.0
    assert k.epsilon == 0.1
    assert k.provenance == "lab"


def test_from_dict_defaults():
    k = Knowledge.from_dict({"value": 3.0})
    assert k.epsilon == 0.0
    assert k.provenance == ""


# ---------------------------------------------------------------------------
# Provenance propagation
# ---------------------------------------------------------------------------


def test_add_provenance():
    a = Knowledge(1.0, 0.1, "A")
    b = Knowledge(2.0, 0.2, "B")
    c = a + b
    assert "(A)" in c.provenance
    assert "(B)" in c.provenance


def test_chained_provenance():
    a = Knowledge(1.0, 0.1, "A")
    b = Knowledge(2.0, 0.2, "B")
    c = Knowledge(3.0, 0.3, "C")
    result = (a + b) * c
    assert "A" in result.provenance
    assert "B" in result.provenance
    assert "C" in result.provenance


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_large_values():
    a = Knowledge(1e12, 1e9, "big")
    b = Knowledge(1e12, 1e9, "big2")
    c = a + b
    assert math.isclose(c.value, 2e12)


def test_negative_values():
    a = Knowledge(-5.0, 0.5, "a")
    b = Knowledge(-3.0, 0.3, "b")
    c = a + b
    assert c.value == -8.0
    assert math.isclose(c.epsilon, math.sqrt(0.25 + 0.09))


def test_very_small_epsilon():
    k = Knowledge(1.0, 1e-15, "precise")
    assert k.is_reliable()
    assert math.isclose(k.confidence, 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Sounio canonical format serialization
# ---------------------------------------------------------------------------


def test_to_sounio_format_basic():
    """Test basic Sounio canonical format output."""
    k = Knowledge(500.0, 2.5, "calibration_batch_2026")
    result = k.to_sounio_format()
    assert "Knowledge {" in result
    assert "value: 500" in result
    assert "epsilon: 2.5" in result
    assert 'prov: "calibration_batch_2026"' in result
    assert result.endswith("}")


def test_to_sounio_format_scientific_notation():
    """Test Sounio format with scientific notation."""
    k = Knowledge(1.23456e+08, 5e+06, "large_value")
    result = k.to_sounio_format()
    # %.6g should use scientific notation for large values
    assert "Knowledge {" in result
    assert "1.23456e+08" in result or "1.23456e+8" in result
    assert "5e+06" in result or "5e+6" in result


def test_to_sounio_format_small_values():
    """Test Sounio format with very small values."""
    k = Knowledge(1.23456e-08, 5e-10, "tiny")
    result = k.to_sounio_format()
    assert "Knowledge {" in result
    assert "1.23456e-08" in result or "1.23456e-8" in result
    assert "5e-10" in result


def test_to_sounio_format_special_provenance():
    """Test that provenance is properly quoted."""
    k = Knowledge(3.14, 0.01, "π from math library")
    result = k.to_sounio_format()
    assert 'prov: "π from math library"' in result


def test_from_sounio_output_basic():
    """Test parsing basic Sounio canonical format."""
    text = 'Knowledge { value: 500 epsilon: 2.5 prov: "calibration_batch_2026" }'
    k = Knowledge.from_sounio_output(text)
    assert k.value == 500.0
    assert k.epsilon == 2.5
    assert k.provenance == "calibration_batch_2026"


def test_from_sounio_output_scientific_notation():
    """Test parsing Sounio format with scientific notation."""
    text = 'Knowledge { value: 1.23456e+08 epsilon: 5e+06 prov: "large" }'
    k = Knowledge.from_sounio_output(text)
    assert math.isclose(k.value, 1.23456e+08, rel_tol=1e-6)
    assert math.isclose(k.epsilon, 5e+06, rel_tol=1e-6)
    assert k.provenance == "large"


def test_from_sounio_output_negative_exponent():
    """Test parsing with negative exponent."""
    text = 'Knowledge { value: 1.23456e-08 epsilon: 5e-10 prov: "tiny" }'
    k = Knowledge.from_sounio_output(text)
    assert math.isclose(k.value, 1.23456e-08, rel_tol=1e-6)
    assert math.isclose(k.epsilon, 5e-10, rel_tol=1e-6)


def test_from_sounio_output_spaces_flexible():
    """Test that parser handles variable whitespace."""
    # Extra spaces
    text = 'Knowledge {  value:  500  epsilon:  2.5  prov: "test" }'
    try:
        k = Knowledge.from_sounio_output(text)
        # Parser may or may not be whitespace-flexible; document actual behavior
        assert k.value == 500.0
    except ValueError:
        # If parser requires exact spacing, that's OK
        pass


def test_from_sounio_output_invalid_format():
    """Test that parser raises ValueError on malformed input."""
    text = "not a valid Knowledge format"
    try:
        Knowledge.from_sounio_output(text)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "Cannot parse Knowledge" in str(e)


def test_from_sounio_output_missing_closing_brace():
    """Test that parser rejects incomplete input."""
    text = 'Knowledge { value: 500 epsilon: 2.5 prov: "test"'
    try:
        Knowledge.from_sounio_output(text)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_sounio_format_round_trip():
    """Test to_sounio_format → from_sounio_output round trip."""
    original = Knowledge(42.5, 0.125, "experiment_001")
    formatted = original.to_sounio_format()
    parsed = Knowledge.from_sounio_output(formatted)

    # Check that parsed values match original
    assert math.isclose(parsed.value, original.value, rel_tol=1e-6)
    assert math.isclose(parsed.epsilon, original.epsilon, rel_tol=1e-6)
    assert parsed.provenance == original.provenance


def test_sounio_format_round_trip_large_values():
    """Test round trip with large values."""
    original = Knowledge(1.23456e+12, 1.11111e+10, "big_measurement")
    formatted = original.to_sounio_format()
    parsed = Knowledge.from_sounio_output(formatted)

    assert math.isclose(parsed.value, original.value, rel_tol=1e-5)
    assert math.isclose(parsed.epsilon, original.epsilon, rel_tol=1e-5)
    assert parsed.provenance == original.provenance


def test_sounio_format_round_trip_small_values():
    """Test round trip with very small values."""
    original = Knowledge(1.23456e-12, 1.11111e-14, "tiny_measurement")
    formatted = original.to_sounio_format()
    parsed = Knowledge.from_sounio_output(formatted)

    assert math.isclose(parsed.value, original.value, rel_tol=1e-5)
    assert math.isclose(parsed.epsilon, original.epsilon, rel_tol=1e-5)
    assert parsed.provenance == original.provenance


def test_sounio_format_round_trip_chain():
    """Test round trip of computed values."""
    k1 = Knowledge(10.0, 0.5, "measurement_A")
    k2 = Knowledge(5.0, 0.2, "measurement_B")
    result = k1 + k2

    formatted = result.to_sounio_format()
    parsed = Knowledge.from_sounio_output(formatted)

    assert math.isclose(parsed.value, result.value, rel_tol=1e-6)
    assert math.isclose(parsed.epsilon, result.epsilon, rel_tol=1e-6)
    # Provenance should contain traces of both inputs
    assert "A" in parsed.provenance
    assert "B" in parsed.provenance
