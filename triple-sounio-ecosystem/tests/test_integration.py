"""Integration tests for Triple Sounio Ecosystem

Tests Knowledge round-trip, pipeline output parsing, and GUM arithmetic.
"""

import os
import math
import re
import sys
from pathlib import Path

import pytest

# Add sounio-py to path
SOUNIO_PY = Path(__file__).parent.parent / "sounio-py" / "python"
if SOUNIO_PY.exists():
    sys.path.insert(0, str(SOUNIO_PY))

from sounio import Knowledge, SounioExecutor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def executor():
    """Create a SounioExecutor with proper environment setup."""
    souc_path = os.environ.get("SOUC")
    if not souc_path:
        souc_path = Path(__file__).parent.parent.parent / "artifacts" / "omega" / "souc-bin" / "souc-linux-x86_64-jit"

    stdlib_path = os.environ.get("SOUNIO_STDLIB_PATH")
    if not stdlib_path:
        stdlib_path = Path(__file__).parent.parent.parent / "stdlib"

    if not Path(souc_path).exists():
        pytest.skip(f"souc binary not found at {souc_path}")

    return SounioExecutor(souc_path=str(souc_path), stdlib_path=str(stdlib_path))


@pytest.fixture
def pipeline_path():
    """Path to the full_pipeline.sio file."""
    path = Path(__file__).parent.parent / "drug-discovery" / "examples" / "full_pipeline.sio"
    if not path.exists():
        pytest.skip(f"Pipeline file not found at {path}")
    return str(path)


# ============================================================================
# Tests
# ============================================================================

class TestKnowledgeRoundTrip:
    """Test Knowledge serialization and parsing."""

    def test_knowledge_repr_format(self):
        """Verify Knowledge repr matches canonical format."""
        k = Knowledge(42.0, 0.1, "test_source")
        repr_str = repr(k)

        # Should be: Knowledge { value: 42.000 epsilon: 0.100 prov: "test_source" }
        assert "Knowledge {" in repr_str
        assert "value: 42.000" in repr_str
        assert "epsilon: 0.100" in repr_str
        assert 'prov: "test_source"' in repr_str

    def test_knowledge_round_trip(self):
        """Create Knowledge, get repr, parse back, verify."""
        original = Knowledge(42.0, 0.1, "test")
        repr_str = repr(original)

        # Parse using the regex from _executor
        pattern = r'Knowledge\s*\{\s*value:\s*([0-9eE+\-.]+)\s+epsilon:\s*([0-9eE+\-.]+)\s+prov:\s*"([^"]*)"\s*\}'
        match = re.search(pattern, repr_str)

        assert match is not None, f"Regex failed to match: {repr_str}"
        value, epsilon, prov = match.groups()

        # Reconstruct
        parsed = Knowledge(float(value), float(epsilon), prov)

        # Verify
        assert abs(parsed.value - original.value) < 1e-6
        assert abs(parsed.epsilon - original.epsilon) < 1e-6
        assert parsed.provenance == original.provenance

    def test_canonical_format_match(self):
        """Verify regex in _executor matches actual Knowledge repr."""
        k = Knowledge(123.456, 2.789, "lipinski_screen")
        repr_str = repr(k)

        # This is the regex from _executor.py
        pattern = re.compile(
            r'Knowledge\s*\{\s*value:\s*([0-9eE+\-.]+)\s+'
            r'epsilon:\s*([0-9eE+\-.]+)\s+'
            r'prov:\s*"([^"]*)"\s*\}'
        )

        match = pattern.search(repr_str)
        assert match is not None, f"Executor regex failed on: {repr_str}"

        value, epsilon, prov = match.groups()
        assert abs(float(value) - 123.456) < 0.001
        assert abs(float(epsilon) - 2.789) < 0.001
        assert prov == "lipinski_screen"

    def test_knowledge_serialization(self):
        """Test to_dict/from_dict round-trip."""
        original = Knowledge(99.0, 0.5, "measurement")
        data = original.to_dict()

        reconstructed = Knowledge.from_dict(data)

        assert reconstructed.value == original.value
        assert reconstructed.epsilon == original.epsilon
        assert reconstructed.provenance == original.provenance


class TestPipelineOutputs:
    """Test parsing outputs from the drug-discovery pipeline."""

    def test_pipeline_outputs_knowledge(self, executor, pipeline_path):
        """Run full_pipeline.sio, verify it outputs ≥5 Knowledge values."""
        result = executor.run_file(pipeline_path, timeout=60)

        assert result.ok, f"Pipeline failed: {result.stderr}"
        assert len(result.knowledge_values) >= 5, \
            f"Expected ≥5 Knowledge values, got {len(result.knowledge_values)}"

    def test_pipeline_knowledge_values_count(self, executor, pipeline_path):
        """Verify all 9 expected Knowledge values are present."""
        result = executor.run_file(pipeline_path, timeout=60)

        assert result.ok, f"Pipeline failed: {result.stderr}"
        # Pipeline outputs exactly 9 Knowledge values
        assert len(result.knowledge_values) == 9, \
            f"Expected 9 Knowledge values, got {len(result.knowledge_values)}"

    def test_pipeline_knowledge_values_parseable(self, executor, pipeline_path):
        """Verify all parsed Knowledge values are valid."""
        result = executor.run_file(pipeline_path, timeout=60)

        assert result.ok
        for i, k in enumerate(result.knowledge_values):
            assert isinstance(k, Knowledge), f"Value {i} is not Knowledge: {type(k)}"
            assert isinstance(k.value, float), f"Value {i} value is not float"
            assert isinstance(k.epsilon, float), f"Value {i} epsilon is not float"
            assert isinstance(k.provenance, str), f"Value {i} provenance is not str"

    def test_pipeline_knowledge_provenance(self, executor, pipeline_path):
        """Verify Knowledge values have expected provenances."""
        result = executor.run_file(pipeline_path, timeout=60)

        assert result.ok
        expected_provs = [
            "lipinski_screen",
            "pk_half_life",
            "pk_tmax",
            "pk_cmax",
            "pk_auc",
            "trial_efficacy",
            "trial_adverse",
            "therapeutic_index",
            "pipeline_decision",
        ]

        actual_provs = [k.provenance for k in result.knowledge_values]
        assert actual_provs == expected_provs, \
            f"Provenance mismatch:\n  Expected: {expected_provs}\n  Got: {actual_provs}"


class TestKnowledgeArithmetic:
    """Test GUM uncertainty propagation in Knowledge arithmetic."""

    def test_knowledge_addition_propagates_uncertainty(self):
        """Verify k1 + k2 has correct uncertainty: sqrt(ε1² + ε2²)."""
        k1 = Knowledge(10.0, 0.5, "source1")
        k2 = Knowledge(20.0, 0.3, "source2")

        result = k1 + k2

        # Addition: ε = sqrt(ε1² + ε2²)
        expected_eps = math.sqrt(0.5**2 + 0.3**2)

        assert result.value == 30.0
        assert abs(result.epsilon - expected_eps) < 1e-10
        assert "(source1)+(source2)" in result.provenance

    def test_knowledge_subtraction_propagates_uncertainty(self):
        """Verify k1 - k2 has correct uncertainty: sqrt(ε1² + ε2²)."""
        k1 = Knowledge(50.0, 0.8, "measurement_a")
        k2 = Knowledge(15.0, 0.2, "measurement_b")

        result = k1 - k2

        expected_eps = math.sqrt(0.8**2 + 0.2**2)

        assert result.value == 35.0
        assert abs(result.epsilon - expected_eps) < 1e-10

    def test_knowledge_multiplication_propagates_uncertainty(self):
        """Verify k1 * k2 uses relative uncertainty: ε = |v| * sqrt((ε1/v1)² + (ε2/v2)²)."""
        k1 = Knowledge(2.0, 0.1, "measure1")  # rel_unc = 0.05
        k2 = Knowledge(3.0, 0.15, "measure2")  # rel_unc = 0.05

        result = k1 * k2

        # Relative uncertainties: 0.1/2 = 0.05, 0.15/3 = 0.05
        # Result: v = 6.0, ε = 6.0 * sqrt(0.05² + 0.05²) ≈ 0.424
        rel_eps_squared = (0.1/2.0)**2 + (0.15/3.0)**2
        expected_eps = 6.0 * math.sqrt(rel_eps_squared)

        assert result.value == 6.0
        assert abs(result.epsilon - expected_eps) < 1e-10

    def test_knowledge_division_propagates_uncertainty(self):
        """Verify k1 / k2 uses relative uncertainty."""
        k1 = Knowledge(10.0, 0.5, "numerator")
        k2 = Knowledge(2.0, 0.1, "denominator")

        result = k1 / k2

        # Result: v = 5.0, ε = 5.0 * sqrt((0.5/10)² + (0.1/2)²)
        rel_eps_squared = (0.5/10.0)**2 + (0.1/2.0)**2
        expected_eps = 5.0 * math.sqrt(rel_eps_squared)

        assert result.value == 5.0
        assert abs(result.epsilon - expected_eps) < 1e-10

    def test_knowledge_scalar_multiplication(self):
        """Verify k * scalar: ε = |factor| * εa."""
        k = Knowledge(7.0, 0.2, "source")

        result = k * 3.0

        assert result.value == 21.0
        assert result.epsilon == 0.6
        assert result.provenance == "source"

    def test_knowledge_arithmetic_preserves_provenance_chain(self):
        """Verify provenance tracks operations."""
        k1 = Knowledge(5.0, 0.1, "meas_a")
        k2 = Knowledge(10.0, 0.2, "meas_b")
        k3 = Knowledge(2.0, 0.05, "meas_c")

        result = (k1 + k2) * k3

        # Provenance should show chain
        assert "meas_a" in result.provenance or "meas_b" in result.provenance
        assert "meas_c" in result.provenance


class TestExecutorRegex:
    """Test the _KNOWLEDGE_RE regex directly."""

    def test_regex_matches_simple_knowledge(self):
        """Verify regex matches simple Knowledge format."""
        from sounio._executor import _KNOWLEDGE_RE

        text = 'Knowledge { value: 42.000 epsilon: 0.850 prov: "test" }'
        matches = _KNOWLEDGE_RE.findall(text)

        assert len(matches) == 1
        value, epsilon, prov = matches[0]
        assert float(value) == 42.0
        assert float(epsilon) == 0.85
        assert prov == "test"

    def test_regex_matches_scientific_notation(self):
        """Verify regex handles scientific notation."""
        from sounio._executor import _KNOWLEDGE_RE

        text = 'Knowledge { value: 1.234e-5 epsilon: 5.678e-3 prov: "scientific" }'
        matches = _KNOWLEDGE_RE.findall(text)

        assert len(matches) == 1
        value, epsilon, prov = matches[0]
        assert abs(float(value) - 1.234e-5) < 1e-10
        assert abs(float(epsilon) - 5.678e-3) < 1e-10

    def test_regex_extracts_multiple_knowledge(self):
        """Verify regex extracts all Knowledge values from output."""
        from sounio._executor import _KNOWLEDGE_RE

        text = """
        Knowledge { value: 1.000 epsilon: 0.100 prov: "prov1" }
        Some other output
        Knowledge { value: 2.000 epsilon: 0.200 prov: "prov2" }
        Knowledge { value: 3.000 epsilon: 0.300 prov: "prov3" }
        """

        matches = _KNOWLEDGE_RE.findall(text)
        assert len(matches) == 3


class TestKnowledgeProperties:
    """Test Knowledge computed properties."""

    def test_relative_uncertainty(self):
        """Verify relative_uncertainty = epsilon / |value|."""
        k = Knowledge(100.0, 5.0, "test")
        assert abs(k.relative_uncertainty - 0.05) < 1e-10

    def test_confidence(self):
        """Verify confidence = 1 - relative_uncertainty."""
        k = Knowledge(100.0, 10.0, "test")
        assert abs(k.confidence - 0.9) < 1e-10

    def test_is_reliable(self):
        """Verify is_reliable threshold."""
        k_good = Knowledge(100.0, 2.0, "test")  # rel_unc = 0.02
        k_bad = Knowledge(100.0, 10.0, "test")  # rel_unc = 0.10

        assert k_good.is_reliable(threshold=0.05)
        assert not k_bad.is_reliable(threshold=0.05)


# ============================================================================
# Integration Test (end-to-end)
# ============================================================================

def test_integration_full_pipeline(executor, pipeline_path):
    """Full integration: run pipeline and verify all Knowledge values."""
    result = executor.run_file(pipeline_path, timeout=60)

    assert result.ok, f"Pipeline failed with stderr: {result.stderr}"
    assert len(result.knowledge_values) == 9

    # Verify each value is reasonable for a drug discovery pipeline
    ks = result.knowledge_values

    # Screening confidence should be < 1
    assert ks[0].value <= 1.0
    assert ks[0].provenance == "lipinski_screen"

    # Half-life should be positive
    assert ks[1].value > 0.0
    assert ks[1].provenance == "pk_half_life"

    # AUC should be positive
    assert ks[4].value > 0.0
    assert ks[4].provenance == "pk_auc"

    # Trial efficacy should be in [0, 1]
    assert 0.0 <= ks[5].value <= 1.0
    assert ks[5].provenance == "trial_efficacy"

    # Therapeutic index should be positive
    assert ks[7].value > 0.0
    assert ks[7].provenance == "therapeutic_index"
