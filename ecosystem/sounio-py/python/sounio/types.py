"""Domain types for the drug-discovery / pharmacokinetics pipeline.

All numeric fields use Knowledge for epistemic tracking.  Dataclasses are used
so that instances are easy to create in both production code and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .knowledge import Knowledge


# ---------------------------------------------------------------------------
# Molecular screening
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    """A small molecule candidate with physicochemical properties.

    Attributes
    ----------
    name : str
        Human-readable compound name or identifier.
    smiles : str
        SMILES string (simplified molecular-input line-entry system).
    molecular_weight : Knowledge
        Molecular weight in Daltons with uncertainty.
    logp : Knowledge
        Lipophilicity (logP) with uncertainty.
    hbd : int
        Number of hydrogen-bond donors.
    hba : int
        Number of hydrogen-bond acceptors.
    passes_lipinski : bool or None
        True when Ro5 criteria are satisfied, None when not yet evaluated.
    """

    name: str
    smiles: str
    molecular_weight: Knowledge
    logp: Knowledge
    hbd: int = 0
    hba: int = 0
    passes_lipinski: Optional[bool] = None

    def lipinski_rule_of_five(self) -> bool:
        """Evaluate Lipinski's Rule of Five and set passes_lipinski."""
        self.passes_lipinski = (
            self.molecular_weight.value <= 500.0
            and self.logp.value <= 5.0
            and self.hbd <= 5
            and self.hba <= 10
        )
        return self.passes_lipinski


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------


@dataclass
class PKParameters:
    """Fitted pharmacokinetic parameters for a molecule–patient pair.

    Attributes
    ----------
    clearance : Knowledge
        Total body clearance (L/h).
    volume_dist : Knowledge
        Volume of distribution (L).
    half_life : Knowledge
        Elimination half-life (h) — derived as ln(2)·Vd/CL.
    bioavailability : Knowledge
        Oral bioavailability as a fraction in [0, 1].
    """

    clearance: Knowledge
    volume_dist: Knowledge
    half_life: Knowledge
    bioavailability: Knowledge


# ---------------------------------------------------------------------------
# Patient / dosing
# ---------------------------------------------------------------------------


@dataclass
class PatientData:
    """Demographic and dosing information for a study subject.

    Attributes
    ----------
    patient_id : str
        Unique subject identifier.
    weight : Knowledge
        Body weight in kg.
    age : int
        Age in years.
    dose : Knowledge
        Administered dose in mg.
    pk_params : PKParameters or None
        Fitted PK parameters when available.
    """

    patient_id: str
    weight: Knowledge
    age: int
    dose: Knowledge
    pk_params: Optional[PKParameters] = None


# ---------------------------------------------------------------------------
# Simulation output
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Aggregate outcome of a virtual population simulation.

    Attributes
    ----------
    efficacy_rate : Knowledge
        Fraction of virtual patients achieving target exposure.
    adverse_rate : Knowledge
        Fraction of virtual patients experiencing adverse events.
    therapeutic_index : Knowledge
        Ratio efficacy_rate / adverse_rate (higher is better).
    confidence : float
        Overall confidence score in [0, 1] (e.g. based on data quality).
    n_patients : int
        Number of virtual patients simulated.
    """

    efficacy_rate: Knowledge
    adverse_rate: Knowledge
    therapeutic_index: Knowledge
    confidence: float
    n_patients: int = 0


@dataclass
class ScreeningResult:
    """Output of a single molecule screening step."""

    molecule: Molecule
    passed: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# End-to-end pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Top-level result returned by the drug-discovery pipeline.

    Attributes
    ----------
    molecules_screened : int
        Total number of compounds entered the pipeline.
    molecules_passed : int
        Compounds that passed all screening filters.
    pk_fitted : int
        Molecules with successfully fitted PK parameters.
    simulation : SimulationResult or None
        Population simulation result when available.
    provenance_chain : list[str]
        Ordered list of provenance labels accumulated during the run.
    stdout : str
        Raw stdout from the souc process (if run via executor).
    exit_code : int
        souc process exit code (0 = success).
    """

    molecules_screened: int
    molecules_passed: int
    pk_fitted: int
    simulation: Optional[SimulationResult] = None
    knowledge_values: List[Knowledge] = field(default_factory=list)
    provenance_chain: List[str] = field(default_factory=list)
    stdout: str = ""
    exit_code: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def summary(self) -> str:
        lines = [
            f"Pipeline result:",
            f"  screened={self.molecules_screened}  passed={self.molecules_passed}"
            f"  pk_fitted={self.pk_fitted}",
        ]
        if self.simulation is not None:
            s = self.simulation
            lines.append(
                f"  efficacy={s.efficacy_rate.value:.3f}±{s.efficacy_rate.epsilon:.3f}"
                f"  adverse={s.adverse_rate.value:.3f}±{s.adverse_rate.epsilon:.3f}"
                f"  TI={s.therapeutic_index.value:.2f}  n={s.n_patients}"
            )
        if self.provenance_chain:
            lines.append(f"  provenance: {' -> '.join(self.provenance_chain)}")
        return "\n".join(lines)
