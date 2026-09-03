"""Tests for DrugDiscoveryPipeline high-level API.

Run with:
    cd ecosystem/sounio-py
    PYTHONPATH=python \
    SOUC=../../bin/souc \
    SOUNIO_STDLIB_PATH=../../stdlib \
    python3 -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import os
import pytest

from sounio._executor import ExecutionResult, SounioExecutor
from sounio.knowledge import Knowledge
from sounio.pipeline import DrugDiscoveryPipeline
from sounio.types import PipelineResult, SimulationResult


# ---------------------------------------------------------------------------
# Fixtures / skip guard
# ---------------------------------------------------------------------------

def _souc_available() -> bool:
    """Return True when the souc binary is resolvable."""
    for var in ("SOUC", "SOUNIO_SOUC_PATH"):
        val = os.environ.get(var)
        if val and os.path.isfile(val):
            return True
    # Repo-relative fallbacks used by SounioExecutor
    for rel in SounioExecutor._DEFAULT_SOUC_CANDIDATES:
        if os.path.isfile(rel):
            return True
    return False


SKIP_NO_SOUC = pytest.mark.skipif(
    not _souc_available(),
    reason="souc binary not found; set SOUC= to run integration tests",
)


def _pipeline_dir() -> str:
    """Resolve drug-discovery directory for tests."""
    env = os.environ.get("SOUNIO_DRUG_DISCOVERY_PATH", "")
    if env and os.path.isdir(env):
        return env
    # Relative from tests/ → ../../drug-discovery
    here = os.path.dirname(__file__)
    candidate = os.path.abspath(os.path.join(here, "..", "..", "drug-discovery"))
    if os.path.isdir(candidate):
        return candidate
    pytest.skip("drug-discovery directory not found")


@pytest.fixture
def pipeline() -> DrugDiscoveryPipeline:
    """Return a DrugDiscoveryPipeline with explicitly resolved pipeline_dir."""
    return DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())


# ---------------------------------------------------------------------------
# Unit tests (no souc needed)
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_pipeline_finds_dir(self) -> None:
        """_find_pipeline_dir must locate drug-discovery without explicit path."""
        pd = _pipeline_dir()
        # Construct without pipeline_dir; pass it via env so _find works
        old = os.environ.get("SOUNIO_DRUG_DISCOVERY_PATH")
        os.environ["SOUNIO_DRUG_DISCOVERY_PATH"] = pd
        try:
            p = DrugDiscoveryPipeline()
            assert os.path.isdir(p.pipeline_dir)
            assert "drug-discovery" in p.pipeline_dir
        finally:
            if old is None:
                del os.environ["SOUNIO_DRUG_DISCOVERY_PATH"]
            else:
                os.environ["SOUNIO_DRUG_DISCOVERY_PATH"] = old

    def test_pipeline_repr(self, pipeline: DrugDiscoveryPipeline) -> None:
        r = repr(pipeline)
        assert "DrugDiscoveryPipeline" in r
        assert "drug-discovery" in r

    def test_available_stages(self, pipeline: DrugDiscoveryPipeline) -> None:
        """All three stage files must be present in drug-discovery/examples/."""
        stages = pipeline.available_stages
        assert "screening" in stages, f"stages={stages}"
        assert "pkpd" in stages, f"stages={stages}"
        assert "full_pipeline" in stages, f"stages={stages}"


class TestParsePipelineResult:
    """Test _parse_pipeline_result with a mocked ExecutionResult."""

    # Minimal full_pipeline.sio-like stdout that contains all sections
    _FULL_STDOUT = """\
========================================
  DRUG DISCOVERY PIPELINE
  Full End-to-End Demo
========================================

STAGE 1: Virtual Screening
--------------------------
  MW  = 129.160 Da, LogP = -1.430, HBD = 2, HBA = 5
  Knowledge { value: 1.000 epsilon: 0.940 prov: "lipinski_screen" }
  Result: PASS

STAGE 2: PK/PD Modeling
-----------------------
  Knowledge { value: 4.620 epsilon: 0.767 prov: "pk_half_life" }
  Knowledge { value: 1.538 epsilon: 0.740 prov: "pk_tmax" }
  Knowledge { value: 7.230 epsilon: 0.600 prov: "pk_cmax" }
  Knowledge { value: 56.667 epsilon: 0.810 prov: "pk_auc" }

STAGE 3: Clinical Trial Simulation
-----------------------------------
  Patients: 100
  Knowledge { value: 0.750 epsilon: 0.950 prov: "trial_efficacy" }
  Knowledge { value: 0.050 epsilon: 0.950 prov: "trial_adverse" }
  Knowledge { value: 4.000 epsilon: 0.900 prov: "therapeutic_index" }

FINAL DECISION
--------------
  PROCEED -- advance to Phase II
  Knowledge { value: 0.680 epsilon: 0.680 prov: "pipeline_decision" }

========================================
  Pipeline complete.
========================================
"""

    def _make_result(self, stdout: str, exit_code: int = 0) -> ExecutionResult:
        from sounio._executor import _KNOWLEDGE_RE
        kvs = [
            Knowledge(float(v), float(e), p)
            for v, e, p in _KNOWLEDGE_RE.findall(stdout)
        ]
        return ExecutionResult(
            stdout=stdout,
            stderr="",
            exit_code=exit_code,
            knowledge_values=kvs,
        )

    def test_parse_stages_present(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        raw = self._make_result(self._FULL_STDOUT)
        result = pipeline._parse_pipeline_result(raw)

        assert result.molecules_screened == 1
        assert result.molecules_passed == 1
        assert result.pk_fitted == 1

    def test_parse_simulation_fields(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        raw = self._make_result(self._FULL_STDOUT)
        result = pipeline._parse_pipeline_result(raw)

        assert result.simulation is not None
        sim = result.simulation
        assert abs(sim.efficacy_rate.value - 0.75) < 1e-6
        assert abs(sim.adverse_rate.value - 0.05) < 1e-6
        assert abs(sim.therapeutic_index.value - 4.0) < 1e-6
        assert sim.n_patients == 100
        assert abs(sim.confidence - 0.680) < 1e-3

    def test_parse_provenance_chain(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        raw = self._make_result(self._FULL_STDOUT)
        result = pipeline._parse_pipeline_result(raw)

        assert "lipinski_screen" in result.provenance_chain
        assert "pk_half_life" in result.provenance_chain
        assert "trial_efficacy" in result.provenance_chain
        assert "pipeline_decision" in result.provenance_chain

    def test_parse_fail_exit_code(self) -> None:
        stdout_fail = self._FULL_STDOUT.replace("PASS", "FAIL").replace("STAGE 2:", "NOPE:")
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        raw = self._make_result(stdout_fail, exit_code=1)
        result = pipeline._parse_pipeline_result(raw)

        assert result.exit_code == 1
        assert not result.ok

    def test_parse_empty_stdout(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        raw = self._make_result("", exit_code=0)
        result = pipeline._parse_pipeline_result(raw)

        assert result.molecules_screened == 0
        assert result.molecules_passed == 0
        assert result.simulation is None
        assert result.provenance_chain == []


class TestGenerateReportUnit:
    """Test generate_report without running souc."""

    def _make_pipeline_result(self) -> PipelineResult:
        from sounio.types import SimulationResult
        sim = SimulationResult(
            efficacy_rate=Knowledge(0.75, 0.05, "trial_efficacy"),
            adverse_rate=Knowledge(0.05, 0.01, "trial_adverse"),
            therapeutic_index=Knowledge(4.0, 0.5, "therapeutic_index"),
            confidence=0.68,
            n_patients=100,
        )
        return PipelineResult(
            molecules_screened=1,
            molecules_passed=1,
            pk_fitted=1,
            simulation=sim,
            knowledge_values=[
                Knowledge(1.0, 0.94, "lipinski_screen"),
                Knowledge(4.62, 0.767, "pk_half_life"),
                Knowledge(0.75, 0.95, "trial_efficacy")
            ],
            provenance_chain=["lipinski_screen", "pk_half_life", "trial_efficacy"],
            stdout="STAGE 1\nSTAGE 2\nKnowledge { value: 4.620 epsilon: 0.767 prov: \"pk_half_life\" }",
            exit_code=0,
        )

    def test_generate_report_markdown(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        result = self._make_pipeline_result()
        report = pipeline.generate_report(result, title="Test Report")

        assert "# Test Report" in report
        assert "Pipeline Summary" in report
        assert "Molecules screened" in report
        assert "Efficacy" in report or "efficacy" in report

    def test_generate_report_provenance(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        result = self._make_pipeline_result()
        report = pipeline.generate_report(result)

        assert "lipinski_screen" in report
        assert "pk_half_life" in report

    def test_generate_report_pk_table(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        result = self._make_pipeline_result()
        report = pipeline.generate_report(result)

        # pk_half_life should appear in the PK table
        assert "pk_half_life" in report

    def test_generate_report_latex(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        result = self._make_pipeline_result()
        report = pipeline.generate_report(result, fmt="latex")

        assert r"\documentclass" in report
        assert r"\section" in report

    def test_generate_report_author(self) -> None:
        pipeline = DrugDiscoveryPipeline(pipeline_dir=_pipeline_dir())
        result = self._make_pipeline_result()
        report = pipeline.generate_report(result, author="Dr. Sounio")

        assert "Dr. Sounio" in report


# ---------------------------------------------------------------------------
# Integration tests (require souc binary)
# ---------------------------------------------------------------------------


class TestRunScreeningIntegration:
    @SKIP_NO_SOUC
    def test_run_screening_exits_zero(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_screening(timeout=60)
        assert raw.exit_code == 0, f"stderr: {raw.stderr}"

    @SKIP_NO_SOUC
    def test_run_screening_has_knowledge_values(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_screening(timeout=60)
        assert len(raw.knowledge_values) > 0

    @SKIP_NO_SOUC
    def test_run_screening_lipinski_prov(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_screening(timeout=60)
        provs = {kv.provenance for kv in raw.knowledge_values}
        assert "lipinski_screen" in provs

    @SKIP_NO_SOUC
    def test_parse_screening_result(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_screening(timeout=60)
        parsed = pipeline.parse_screening_result(raw)
        assert parsed["total_screened"] == 5
        assert 0 <= parsed["total_passed"] <= 5


class TestRunPKPDIntegration:
    @SKIP_NO_SOUC
    def test_run_pkpd_exits_zero(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_pkpd(timeout=60)
        assert raw.exit_code == 0, f"stderr: {raw.stderr}"

    @SKIP_NO_SOUC
    def test_run_pkpd_has_half_life(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_pkpd(timeout=60)
        parsed = pipeline.parse_pkpd_result(raw)
        assert "pk_half_life" in parsed
        hl = parsed["pk_half_life"]
        # half-life should be positive
        assert hl.value > 0.0

    @SKIP_NO_SOUC
    def test_run_pkpd_has_all_pk_keys(self, pipeline: DrugDiscoveryPipeline) -> None:
        raw = pipeline.run_pkpd(timeout=60)
        parsed = pipeline.parse_pkpd_result(raw)
        expected_keys = {"pk_half_life", "pk_tmax", "pk_cmax", "pk_auc"}
        missing = expected_keys - parsed.keys()
        assert not missing, f"Missing PK keys: {missing}"


class TestRunFullIntegration:
    @SKIP_NO_SOUC
    def test_run_full_returns_pipeline_result(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        assert isinstance(result, PipelineResult)

    @SKIP_NO_SOUC
    def test_run_full_exit_code(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        assert result.exit_code == 0, f"stdout: {result.stdout[:500]}"

    @SKIP_NO_SOUC
    def test_run_full_molecules_screened(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        # full_pipeline screens one candidate
        assert result.molecules_screened == 1

    @SKIP_NO_SOUC
    def test_run_full_molecules_passed(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        # The metformin-like candidate should pass Lipinski
        assert result.molecules_passed == 1

    @SKIP_NO_SOUC
    def test_run_full_pk_fitted(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        assert result.pk_fitted == 1

    @SKIP_NO_SOUC
    def test_run_full_simulation_populated(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        assert result.simulation is not None
        sim = result.simulation
        assert 0.0 <= sim.efficacy_rate.value <= 1.0
        assert 0.0 <= sim.adverse_rate.value <= 1.0
        assert sim.n_patients == 100

    @SKIP_NO_SOUC
    def test_run_full_provenance_chain(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        assert len(result.provenance_chain) > 0
        assert "lipinski_screen" in result.provenance_chain
        assert "pk_half_life" in result.provenance_chain

    @SKIP_NO_SOUC
    def test_generate_report_from_live_run(self, pipeline: DrugDiscoveryPipeline) -> None:
        result = pipeline.run_full(timeout=90)
        report = pipeline.generate_report(result, title="Live Pipeline Report")

        assert "# Live Pipeline Report" in report
        assert "Pipeline Summary" in report
        assert "Molecules screened: 1" in report
