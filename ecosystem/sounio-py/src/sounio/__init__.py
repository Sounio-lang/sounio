"""
sounio — Python bindings for epistemic scientific computing.

The first package manager and runtime for trustworthy scientific software.

Quick start::

    from sounio import Knowledge, measure, confidence_gate

    dose = measure(500.0, uncertainty=25.0, unit="mg")
    volume = measure(10.0, uncertainty=0.2, unit="mL")
    concentration = dose / volume  # GUM uncertainty propagation

    if confidence_gate(concentration.value > 45.0,
                       confidence=concentration.confidence,
                       min_confidence=0.90):
        print("Therapeutic range achieved")

"""

from sounio._knowledge import Knowledge, measure, confidence_gate
from sounio._epistemic import EpistemicResult, GUMPropagation
from sounio._compile import compile as compile_sio, run as run_sio

__version__ = "0.1.0"
__all__ = [
    "Knowledge",
    "measure",
    "confidence_gate",
    "EpistemicResult",
    "GUMPropagation",
    "compile_sio",
    "run_sio",
]
