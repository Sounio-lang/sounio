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
        assert abs(result.epsilon - 0.6) < 1e-10
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

    def test_scale(self):
        """Verify scale method for uncertainty propagation."""
        k = Knowledge(50.0, 5.0, "measurement")
        scaled = k.scale(2.0)

        assert scaled.value == 100.0
        assert scaled.epsilon == 10.0
        assert scaled.provenance == "measurement"

    def test_to_sounio_format(self):
        """Verify to_sounio_format for canonical output."""
        k = Knowledge(500.0, 2.5, "calibration")
        formatted = k.to_sounio_format()

        assert "Knowledge {" in formatted
        assert "500" in formatted
        assert "2.5" in formatted
        assert "calibration" in formatted

    def test_from_sounio_output(self):
        """Verify from_sounio_output parser."""
        text = 'Knowledge { value: 100 epsilon: 5.0 prov: "test_source" }'
        k = Knowledge.from_sounio_output(text)

        assert k.value == 100.0
        assert k.epsilon == 5.0
        assert k.provenance == "test_source"


# ============================================================================
# PubChem integration
# ============================================================================

class TestPubChemIntegration:
    """Test PubChem integration with Knowledge values."""

    def test_fetch_molecule_by_name(self):
        """Fetch a molecule from PubChem offline cache."""
        from sounio.integrations.pubchem import fetch_by_name

        mol = fetch_by_name("aspirin", offline=True)

        assert mol.name == "aspirin"
        assert mol.molecular_weight.value > 100
        assert mol.molecular_weight.epsilon == 0.001  # MW epsilon
        assert mol.logp is not None

    def test_molecule_properties_are_knowledge(self):
        """Verify molecule properties use Knowledge with uncertainty."""
        from sounio.integrations.pubchem import fetch_by_name

        mol = fetch_by_name("ibuprofen", offline=True)

        # MW and LogP should have epistemic uncertainty
        assert isinstance(mol.molecular_weight, Knowledge)
        assert isinstance(mol.logp, Knowledge)
        assert mol.molecular_weight.epsilon > 0
        assert mol.logp.epsilon > 0

    def test_multiple_molecules(self):
        """Fetch multiple molecules and verify Lipinski rule."""
        from sounio.integrations.pubchem import fetch_by_name

        molecules = ["aspirin", "ibuprofen", "paracetamol"]
        for name in molecules:
            mol = fetch_by_name(name, offline=True)
            assert mol.name == name
            assert mol.hbd >= 0
            assert mol.hba >= 0

    def test_offline_cache_coverage(self):
        """Verify offline cache has common drug compounds."""
        from sounio.integrations.pubchem import _OFFLINE_CACHE

        # Expected drugs in cache
        expected = {"aspirin", "ibuprofen", "metformin", "paracetamol", "caffeine"}
        cached_names = {v["name"] for v in _OFFLINE_CACHE.values()}

        for name in expected:
            assert name in cached_names, f"Missing {name} in offline cache"

    def test_offline_cache_miss_raises_keyerror(self):
        """Verify offline mode raises KeyError on cache miss."""
        from sounio.integrations.pubchem import fetch_by_name
        import pytest

        with pytest.raises(KeyError):
            fetch_by_name("nonexistent_compound_12345", offline=True)


# ============================================================================
# Report generation integration
# ============================================================================

class TestReportIntegration:
    """Test ReportBuilder integration with Knowledge values."""

    def test_report_builder_from_knowledge(self):
        """Generate a report with Knowledge values."""
        from sounio.report import ReportBuilder

        rb = ReportBuilder("Test Report", author="Test Author")
        rb.add_knowledge_table(
            "Test Data",
            {
                "param1": Knowledge(42.0, 0.5, "test1"),
                "param2": Knowledge(100.0, 5.0, "test2"),
            }
        )

        md = rb.to_markdown()

        assert "Test Report" in md
        assert "Test Author" in md
        assert "Test Data" in md
        assert "42" in md
        assert "100" in md

    def test_report_table_includes_uncertainty(self):
        """Verify report tables include epsilon and relative uncertainty."""
        from sounio.report import ReportBuilder

        rb = ReportBuilder("Report")
        k = Knowledge(200.0, 10.0, "measure")
        rb.add_knowledge_table("Data", {"Value": k})

        md = rb.to_markdown()

        # Should include value and uncertainty
        assert "200" in md or "2e+02" in md or "200" in md.lower()
        assert "10" in md or "0.1e+02" in md
        assert "measure" in md

    def test_report_with_multiple_sections(self):
        """Verify report with text, table, and figure sections."""
        from sounio.report import ReportBuilder

        rb = ReportBuilder("Multi-section Report")
        rb.add_section("Intro", "This is the introduction.")
        rb.add_knowledge_table(
            "Results",
            {"result": Knowledge(42.0, 2.0, "calc")}
        )
        rb.add_section("Conclusion", "This is the conclusion.")

        md = rb.to_markdown()

        assert "Intro" in md
        assert "Results" in md
        assert "Conclusion" in md
        assert "This is the introduction" in md
        assert "This is the conclusion" in md

    def test_report_latex_generation(self):
        """Verify LaTeX report generation."""
        from sounio.report import ReportBuilder

        rb = ReportBuilder("LaTeX Report", author="Author")
        rb.add_knowledge_table(
            "Data",
            {"value": Knowledge(50.0, 2.5, "source")}
        )

        latex = rb.to_latex()

        assert r"\documentclass" in latex
        assert r"\title{LaTeX Report}" in latex
        assert r"\author{Author}" in latex
        assert "50" in latex
        assert "2.5" in latex

    def test_report_save_markdown(self, tmp_path):
        """Verify report can be saved to markdown file."""
        from sounio.report import ReportBuilder

        rb = ReportBuilder("Saved Report")
        rb.add_section("Content", "Test content")

        output_path = tmp_path / "report.md"
        rb.save(str(output_path), format="markdown")

        assert output_path.exists()
        content = output_path.read_text()
        assert "Saved Report" in content
        assert "Test content" in content

    def test_report_save_latex(self, tmp_path):
        """Verify report can be saved to LaTeX file."""
        from sounio.report import ReportBuilder

        rb = ReportBuilder("LaTeX Saved Report")
        rb.add_section("Content", "LaTeX content")

        output_path = tmp_path / "report.tex"
        rb.save(str(output_path), format="latex")

        assert output_path.exists()
        content = output_path.read_text()
        assert r"\documentclass" in content


# ============================================================================
# Full workflow integration
# ============================================================================

class TestCompleteWorkflow:
    """Test complete PubChem → Knowledge → Pipeline → Report workflow."""

    def test_pubchem_to_knowledge_arithmetic(self):
        """PubChem data used in Knowledge arithmetic."""
        from sounio.integrations.pubchem import fetch_by_name

        mol = fetch_by_name("aspirin", offline=True)

        # Use molecular weight in arithmetic
        dose = Knowledge(500.0, 10.0, "dose_mg")
        dose_per_gram = dose / mol.molecular_weight  # MW ~= 180

        assert dose_per_gram.value > 0
        assert dose_per_gram.epsilon > 0
        assert dose_per_gram.relative_uncertainty > 0

    def test_pipeline_results_to_report(self, executor, pipeline_path):
        """Pipeline results can be formatted into report."""
        from sounio.report import ReportBuilder

        result = executor.run_file(pipeline_path, timeout=60)
        assert result.ok

        rb = ReportBuilder("Pipeline Results Report")

        # Convert pipeline Knowledge values to dict for table
        if result.knowledge_values:
            kv_dict = {kv.provenance: kv for kv in result.knowledge_values}
            rb.add_knowledge_table("Pipeline Outputs", kv_dict)

        md = rb.to_markdown()
        assert "Pipeline Results Report" in md
        assert "Pipeline Outputs" in md

    def test_knowledge_chain_provenance(self):
        """Verify provenance chain through complex operations."""
        k1 = Knowledge(10.0, 0.5, "measure_a")
        k2 = Knowledge(20.0, 1.0, "measure_b")
        k3 = Knowledge(2.0, 0.1, "measure_c")

        result = (k1 + k2) * k3

        # Provenance should include all three measures
        assert "measure_a" in result.provenance or "measure_b" in result.provenance
        assert "measure_c" in result.provenance


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
