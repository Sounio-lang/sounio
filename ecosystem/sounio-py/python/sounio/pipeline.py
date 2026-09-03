"""High-level Python API for the Sounio drug discovery pipeline.

Wraps the three stages (virtual screening, PK/PD modeling, clinical trial
simulation) that live in drug-discovery/examples/ as .sio programs and
exposes them through a single Pythonic interface.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .knowledge import Knowledge
from ._executor import SounioExecutor, ExecutionResult
from .types import (
    SimulationResult,
    ScreeningResult,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Detect stage headers emitted by full_pipeline.sio
_STAGE1_RE = re.compile(r"STAGE 1")
_STAGE2_RE = re.compile(r"STAGE 2")
_PASS_RE   = re.compile(r"\bPASS\b")
_FAIL_RE   = re.compile(r"\bFAIL\b")

# Screening demo prints "N / 5 molecules passed"
_SCREENED_RE  = re.compile(r"(\d+)\s*/\s*(\d+)\s*molecules passed")

# full_pipeline.sio prints "Patients: N"
_PATIENTS_RE  = re.compile(r"Patients:\s*(\d+)")


def _kv_map(kvs: List[Knowledge]) -> Dict[str, Knowledge]:
    """Build provenance → Knowledge dict; last write wins on duplicates."""
    result: Dict[str, Knowledge] = {}
    for k in kvs:
        result[k.provenance] = k
    return result


def _zero(prov: str) -> Knowledge:
    return Knowledge(0.0, 0.0, prov)


# ---------------------------------------------------------------------------
# DrugDiscoveryPipeline
# ---------------------------------------------------------------------------


class DrugDiscoveryPipeline:
    """Pythonic interface to the Sounio drug discovery pipeline.

    The pipeline is backed by three Sounio programs:

    * **screening_demo.sio** — virtual screening (Lipinski Rule of Five)
    * **pkpd_demo.sio**      — one-compartment oral PK/PD model
    * **full_pipeline.sio**  — end-to-end: screening → PK/PD → Monte Carlo trial

    Parameters
    ----------
    souc_path : str, optional
        Explicit path to the ``souc`` JIT binary.  Falls back to the
        ``SOUC`` / ``SOUNIO_SOUC_PATH`` environment variables and then to the
        repo-relative default.
    pipeline_dir : str, optional
        Explicit path to the ``drug-discovery`` directory.  Falls back to
        ``SOUNIO_DRUG_DISCOVERY_PATH`` and then to paths relative to this
        file.

    Examples
    --------
    >>> pipeline = DrugDiscoveryPipeline()
    >>> result = pipeline.run_full()
    >>> print(result.summary())

    >>> # Step-by-step
    >>> screening = pipeline.run_screening()
    >>> pk        = pipeline.run_pkpd()
    >>> sim       = pipeline.run_simulation()
    >>> report    = pipeline.generate_report(result)
    """

    def __init__(
        self,
        souc_path: Optional[str] = None,
        pipeline_dir: Optional[str] = None,
    ) -> None:
        self.executor = SounioExecutor(souc_path)
        self.pipeline_dir = (
            pipeline_dir if pipeline_dir else self._find_pipeline_dir()
        )

    # ---- Directory resolution --------------------------------------------

    def _find_pipeline_dir(self) -> str:
        """Locate the drug-discovery directory."""
        here = Path(__file__).resolve().parent
        
        candidates = [
            Path(p) for p in [os.environ.get("SOUNIO_DRUG_DISCOVERY_PATH", "")]
            if p
        ]
        
        # sounio-py/python/sounio/ → ../../../drug-discovery
        candidates.append(here.parent.parent.parent / "drug-discovery")
        # repo root / ecosystem / drug-discovery
        candidates.append(here.parent.parent.parent.parent / "ecosystem" / "drug-discovery")

        for c in candidates:
            if c.is_dir():
                return str(c.resolve())
                
        raise FileNotFoundError(
            "drug-discovery directory not found. "
            "Set SOUNIO_DRUG_DISCOVERY_PATH or pass pipeline_dir=."
        )

    def _example(self, filename: str) -> str:
        return str(Path(self.pipeline_dir) / "examples" / filename)

    # ---- Individual stages -----------------------------------------------

    def run_screening(self, timeout: int = 30) -> ExecutionResult:
        """Run the virtual screening stage (Lipinski Rule of Five).

        Returns
        -------
        ExecutionResult
            Raw executor result including parsed Knowledge values and stdout.
        """
        return self.executor.run_file(self._example("screening_demo.sio"), timeout=timeout)

    def run_pkpd(self, timeout: int = 30) -> ExecutionResult:
        """Run the PK/PD modeling stage (one-compartment oral model).

        Returns
        -------
        ExecutionResult
            Raw executor result.
        """
        return self.executor.run_file(self._example("pkpd_demo.sio"), timeout=timeout)

    def run_simulation(self, timeout: int = 60) -> ExecutionResult:
        """Run the full end-to-end pipeline including Monte Carlo simulation.

        Returns
        -------
        ExecutionResult
            Raw executor result.
        """
        return self.executor.run_file(self._example("full_pipeline.sio"), timeout=timeout)

    # ---- High-level run --------------------------------------------------

    def run_full(self, timeout: int = 60) -> PipelineResult:
        """Run the complete pipeline and return a structured PipelineResult.

        Executes ``full_pipeline.sio``, parses all Knowledge values and
        stage markers from stdout, and returns a :class:`PipelineResult`.

        Parameters
        ----------
        timeout : int
            Maximum seconds to wait (default 60).

        Returns
        -------
        PipelineResult
            Structured pipeline outcome.
        """
        raw = self.executor.run_file(self._example("full_pipeline.sio"), timeout=timeout)
        return self._parse_pipeline_result(raw)

    # ---- Parsing ---------------------------------------------------------

    def _parse_pipeline_result(self, result: ExecutionResult) -> PipelineResult:
        """Convert a raw ExecutionResult into a structured PipelineResult."""
        stdout = result.stdout
        kvs    = result.knowledge_values
        kmap   = _kv_map(kvs)

        # --- Stage presence detection ---
        stage1_present = bool(_STAGE1_RE.search(stdout))
        stage2_present = bool(_STAGE2_RE.search(stdout))

        # Screening: 1 molecule processed in the full pipeline
        molecules_screened = 1 if stage1_present else 0

        # PASS / FAIL for the Lipinski screen
        molecules_passed = 1 if (stage1_present and _PASS_RE.search(stdout)) else 0

        # PK fitted flag
        pk_fitted = 1 if stage2_present else 0

        # --- SimulationResult ---
        sim: Optional[SimulationResult] = None
        if "trial_efficacy" in kmap or "trial_adverse" in kmap:
            efficacy_rate   = kmap.get("trial_efficacy",    _zero("trial_efficacy"))
            adverse_rate    = kmap.get("trial_adverse",     _zero("trial_adverse"))
            therapeutic_idx = kmap.get("therapeutic_index", _zero("therapeutic_index"))
            decision_kv     = kmap.get("pipeline_decision", _zero("pipeline_decision"))

            # n_patients from "Patients: N" line
            m_pat = _PATIENTS_RE.search(stdout)
            n_patients = int(m_pat.group(1)) if m_pat else 100

            sim = SimulationResult(
                efficacy_rate=efficacy_rate,
                adverse_rate=adverse_rate,
                therapeutic_index=therapeutic_idx,
                confidence=decision_kv.value,
                n_patients=n_patients,
            )

        provenance_chain = [kv.provenance for kv in kvs]

        return PipelineResult(
            molecules_screened=molecules_screened,
            molecules_passed=molecules_passed,
            pk_fitted=pk_fitted,
            simulation=sim,
            knowledge_values=kvs,
            provenance_chain=provenance_chain,
            stdout=stdout,
            exit_code=result.exit_code,
        )

    def parse_screening_result(self, result: ExecutionResult) -> dict:
        """Parse screening_demo.sio output into a summary dict.

        Returns
        -------
        dict
            Keys: ``total_screened``, ``total_passed``, ``knowledge_values``.
        """
        stdout = result.stdout
        total_screened = 5  # screening_demo always runs 5 molecules

        total_passed = 0
        m = _SCREENED_RE.search(stdout)
        if m:
            total_passed = int(m.group(1))

        return {
            "total_screened": total_screened,
            "total_passed": total_passed,
            "knowledge_values": result.knowledge_values,
        }

    def parse_pkpd_result(self, result: ExecutionResult) -> Dict[str, Knowledge]:
        """Parse pkpd_demo.sio output into a provenance-keyed dict.

        Returns
        -------
        dict[str, Knowledge]
            E.g. ``{"pk_half_life": Knowledge(...), "pk_cmax": Knowledge(...), ...}``
        """
        return _kv_map(result.knowledge_values)

    # ---- Report generation -----------------------------------------------

    def generate_report(
        self,
        result: PipelineResult,
        title: str = "Drug Discovery Report",
        author: str = "",
        fmt: str = "markdown",
    ) -> str:
        """Generate a reproducible report from pipeline results.

        Parameters
        ----------
        result : PipelineResult
            Output of :meth:`run_full`.
        title : str
            Report title.
        author : str
            Optional author name.
        fmt : str
            Output format: ``"markdown"`` (default) or ``"latex"``.

        Returns
        -------
        str
            Formatted report text.
        """
        from .report import ReportBuilder

        rb = ReportBuilder(title, author=author)

        # Summary section
        rb.add_pipeline_summary(result)

        # PK Knowledge table — extract PK provenance keys
        pk_keys = {
            "pk_half_life", "pk_tmax", "pk_cmax", "pk_auc",
            "bioavailability", "absorption_rate", "elimination_rate",
            "volume_distribution",
        }
        
        # Filter knowledge values for PK parameters
        pk_table = {
            k.provenance: k
            for k in result.knowledge_values
            if k.provenance in pk_keys
        }
        
        # Filter knowledge values for PK parameters
        pk_table = {
            k.provenance: k
            for k in result.knowledge_values
            if k.provenance in pk_keys
        }
        if pk_table:
            rb.add_knowledge_table("PK/PD Parameters", pk_table)

        # Provenance chain
        if result.provenance_chain:
            rb.add_section(
                "Provenance Chain",
                "\n".join(f"- {p}" for p in result.provenance_chain),
            )

        if fmt == "latex":
            return rb.to_latex()
        return rb.to_markdown()

    # ---- Introspection ---------------------------------------------------

    @property
    def available_stages(self) -> List[str]:
        """List pipeline stage names whose .sio files exist on disk.

        Returns
        -------
        list[str]
            Stage names: ``"screening"``, ``"pkpd"``, ``"full_pipeline"``.
        """
        mapping = {
            "screening_demo.sio": "screening",
            "pkpd_demo.sio": "pkpd",
            "full_pipeline.sio": "full_pipeline",
        }
        return [
            stage
            for fname, stage in mapping.items()
            if Path(self._example(fname)).is_file()
        ]

    def __repr__(self) -> str:
        return (
            f"DrugDiscoveryPipeline("
            f"pipeline_dir={self.pipeline_dir!r}, "
            f"executor={self.executor!r})"
        )
