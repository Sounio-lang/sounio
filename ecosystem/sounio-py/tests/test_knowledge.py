"""Tests for sounio._knowledge — Knowledge<T> and GUM propagation."""

import math
import pytest
from sounio._knowledge import Knowledge, measure, confidence_gate


# ============================================================================
# Basic construction
# ============================================================================

def test_knowledge_basic():
    k = Knowledge(10.0, uncertainty=0.5)
    assert k.value == 10.0
    assert k.uncertainty == 0.5
    assert k.confidence == 1.0

def test_measure_constructor():
    k = measure(500.0, uncertainty=25.0, unit="mg", source="HPLC")
    assert k.value == 500.0
    assert k.uncertainty == 25.0
    assert k.unit == "mg"
    assert k.provenance.source == "HPLC"

def test_invalid_uncertainty():
    with pytest.raises(ValueError):
        Knowledge(1.0, uncertainty=-0.1)

def test_invalid_confidence():
    with pytest.raises(ValueError):
        Knowledge(1.0, uncertainty=0.1, confidence=1.5)


# ============================================================================
# GUM arithmetic propagation
# ============================================================================

def test_addition_gum():
    a = Knowledge(10.0, uncertainty=0.5)
    b = Knowledge(20.0, uncertainty=0.3)
    c = a + b
    assert c.value == 30.0
    # u(a+b) = sqrt(0.5² + 0.3²)
    expected_u = math.sqrt(0.5**2 + 0.3**2)
    assert abs(c.uncertainty - expected_u) < 1e-10

def test_subtraction_gum():
    a = Knowledge(20.0, uncertainty=0.4)
    b = Knowledge(5.0, uncertainty=0.2)
    c = a - b
    assert c.value == 15.0
    expected_u = math.sqrt(0.4**2 + 0.2**2)
    assert abs(c.uncertainty - expected_u) < 1e-10

def test_multiplication_gum():
    # dose / volume example from Sounio manifesto
    dose = measure(500.0, uncertainty=25.0, unit="mg")
    volume = measure(10.0, uncertainty=0.2, unit="mL")
    conc = dose / volume
    assert abs(conc.value - 50.0) < 1e-10
    # GUM: δ(A/B)/(A/B) = sqrt((δA/A)² + (δB/B)²)
    rel_u = math.sqrt((25/500)**2 + (0.2/10)**2)
    expected_u = 50.0 * rel_u
    assert abs(conc.uncertainty - expected_u) < 1e-6

def test_division_by_zero():
    a = Knowledge(10.0, uncertainty=0.5)
    b = Knowledge(0.0, uncertainty=0.1)
    with pytest.raises(ZeroDivisionError):
        _ = a / b

def test_scalar_add():
    k = Knowledge(5.0, uncertainty=0.1)
    r = k + 3.0
    assert r.value == 8.0
    assert r.uncertainty == 0.1  # scalar add does not change uncertainty

def test_scalar_mul():
    k = Knowledge(5.0, uncertainty=0.2)
    r = k * 2.0
    assert r.value == 10.0
    assert abs(r.uncertainty - 0.4) < 1e-10


# ============================================================================
# Confidence combination
# ============================================================================

def test_confidence_propagation():
    a = Knowledge(10.0, uncertainty=0.5, confidence=0.95)
    b = Knowledge(20.0, uncertainty=0.3, confidence=0.80)
    c = a + b
    assert c.confidence == 0.80  # minimum of operands


# ============================================================================
# Epistemic helpers
# ============================================================================

def test_cv_percent():
    k = Knowledge(100.0, uncertainty=5.0)
    assert abs(k.cv_percent - 5.0) < 1e-10

def test_prob_gt():
    k = Knowledge(50.0, uncertainty=5.0)
    p_gt_45 = k.prob_gt(45.0)
    assert p_gt_45 > 0.80  # 50 > 45, well within 1 sigma

def test_prob_in_range():
    k = Knowledge(50.0, uncertainty=5.0)
    p = k.prob_in_range(40.0, 60.0)
    assert p > 0.90  # ±2 sigma should contain most probability


# ============================================================================
# confidence_gate
# ============================================================================

def test_gate_passes():
    k = measure(50.0, uncertainty=2.0, confidence=0.95)
    assert confidence_gate(k.value > 45.0, k.confidence, min_confidence=0.90)

def test_gate_fails_low_confidence():
    k = measure(50.0, uncertainty=2.0, confidence=0.70)
    result = confidence_gate(k.value > 45.0, k.confidence,
                             min_confidence=0.90, action="silent")
    assert not result

def test_gate_fails_condition():
    k = measure(40.0, uncertainty=2.0, confidence=0.99)
    assert not confidence_gate(k.value > 45.0, k.confidence, min_confidence=0.90)


# ============================================================================
# Provenance tracking
# ============================================================================

def test_provenance_source():
    k = measure(100.0, uncertainty=5.0, source="clinical_trial_A")
    assert k.provenance.source == "clinical_trial_A"

def test_provenance_derived():
    a = measure(100.0, uncertainty=5.0, source="source_A")
    b = measure(50.0, uncertainty=2.0, source="source_B")
    c = a + b
    assert "source_A" in c.provenance.derived_from
    assert "source_B" in c.provenance.derived_from


# ============================================================================
# repr
# ============================================================================

def test_repr():
    k = Knowledge(10.0, uncertainty=0.5, unit="mg")
    r = repr(k)
    assert "10.0" in r or "10" in r
    assert "0.5" in r
    assert "confidence" in r
