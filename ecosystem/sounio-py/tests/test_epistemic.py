"""Tests for sounio._epistemic — GUMPropagation and EpistemicResult."""

import math
import pytest
from sounio._knowledge import measure
from sounio._epistemic import GUMPropagation, EpistemicResult


def test_gum_propagation_simple():
    """Test GUM propagation for f(x, y) = x * y."""
    gum = GUMPropagation()
    result = gum.propagate(
        fn=lambda x, y: x * y,
        inputs={
            "x": measure(10.0, uncertainty=0.5),
            "y": measure(5.0, uncertainty=0.2),
        },
    )
    assert abs(result.value - 50.0) < 1e-6
    # u(x*y) ≈ sqrt((y*u_x)² + (x*u_y)²) = sqrt((5*0.5)² + (10*0.2)²)
    expected_u = math.sqrt((5.0 * 0.5)**2 + (10.0 * 0.2)**2)
    assert abs(result.uncertainty - expected_u) < 0.01


def test_gum_parallel_resistance():
    """Classic metrology example: parallel resistors R = R1*R2/(R1+R2)."""
    gum = GUMPropagation(step_factor=1e-6)

    def parallel(r1, r2):
        return (r1 * r2) / (r1 + r2)

    result = gum.propagate(
        fn=parallel,
        inputs={
            "R1": measure(100.0, uncertainty=2.0, unit="Ohm"),
            "R2": measure(200.0, uncertainty=5.0, unit="Ohm"),
        },
        output_unit="Ohm",
    )
    expected = parallel(100.0, 200.0)
    assert abs(result.value - expected) < 1e-6
    assert result.uncertainty > 0.0


def test_sensitivity_coefficients():
    gum = GUMPropagation()
    coeffs = gum.sensitivity_coefficients(
        fn=lambda x, y: x + 2.0 * y,
        inputs={
            "x": measure(5.0, uncertainty=0.1),
            "y": measure(3.0, uncertainty=0.1),
        },
    )
    assert abs(coeffs["x"] - 1.0) < 0.001
    assert abs(coeffs["y"] - 2.0) < 0.001


def test_uncertainty_budget():
    gum = GUMPropagation()
    budget = gum.uncertainty_budget(
        fn=lambda x, y: x + y,
        inputs={
            "x": measure(10.0, uncertainty=3.0),
            "y": measure(5.0, uncertainty=1.0),
        },
    )
    # x contributes more: (3²)/(3²+1²) = 90%
    assert budget["x"] > budget["y"]
    assert abs(sum(budget.values()) - 100.0) < 0.1


def test_epistemic_result():
    result = EpistemicResult(
        values={
            "plasma_conc": measure(50.0, uncertainty=5.0, confidence=0.93),
            "brain_conc": measure(12.0, uncertainty=2.0, confidence=0.88),
        },
        epistemic_score=0.91,
    )
    assert result.confidence_summary() == 0.88  # minimum
    assert "plasma_conc" in result.provenance_summary()
    assert result.epistemic_score == 0.91

    d = result.to_dict()
    assert "values" in d
    assert "epistemic_score" in d
