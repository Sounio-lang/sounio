"""
sounio — Python bindings for epistemic scientific computing.

The first package manager and runtime for trustworthy scientific software.

Quick start::

    from sounio import Knowledge, measure, confidence_gate

    dose   = measure(500.0, uncertainty=25.0, unit="mg",  source="HPLC_2025")
    volume = measure(10.0,  uncertainty=0.2,  unit="mL",  source="pipette_A")
    conc   = dose / volume     # GUM first-order propagation
    confidence_gate(conc, 0.90)
    print(conc)                # Knowledge(50 ± 1.27 mg/mL, confidence=0.90)

Backend selection:
    sounio first tries to import the compiled Rust extension
    (`_sounio_native`, built via `maturin develop`). If unavailable,
    it falls back to the pure-Python implementation transparently.
"""

# -- Backend selection --------------------------------------------------
# Try the Rust/PyO3 native extension first (fast, full GUM).
# Fall back to pure-Python implementation when native is not built.
_NATIVE_BACKEND = False
try:
    from sounio._sounio_native import (  # type: ignore[import]
        Knowledge as _NativeKnowledge,
        measure as _native_measure,
        confidence_gate as _native_confidence_gate,
        GUMPropagation as _NativeGUMPropagation,
        EpistemicResult as _NativeEpistemicResult,
        SensitivityCoefficient,
    )
    Knowledge = _NativeKnowledge
    measure = _native_measure
    confidence_gate = _native_confidence_gate
    GUMPropagation = _NativeGUMPropagation
    EpistemicResult = _NativeEpistemicResult
    _NATIVE_BACKEND = True
except ImportError:
    from sounio._knowledge import Knowledge, measure, confidence_gate  # type: ignore[assignment]
    from sounio._epistemic import EpistemicResult, GUMPropagation      # type: ignore[assignment]
    SensitivityCoefficient = None  # available after `maturin develop`

from sounio._compile import compile as compile_sio, run as run_sio

__version__ = "0.2.0"
__backend__ = "native (rust/pyo3)" if _NATIVE_BACKEND else "pure-python"

__all__ = [
    "Knowledge",
    "measure",
    "confidence_gate",
    "EpistemicResult",
    "GUMPropagation",
    "SensitivityCoefficient",
    "compile_sio",
    "run_sio",
    "__backend__",
]
