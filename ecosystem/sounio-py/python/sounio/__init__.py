"""sounio — Python bindings for the Sounio epistemic computing language.

Quick start
-----------
>>> import sounio
>>> x = sounio.Knowledge(36.5, epsilon=0.1, provenance="thermometer")
>>> print(x)
Knowledge(36.500 ± 0.100, prov='thermometer')

>>> y = sounio.Knowledge(1.5, 0.05, "barometer")
>>> z = x + y
>>> print(z)
Knowledge(38.000 ± 0.112, prov='(thermometer)+(barometer)')

>>> # Run inline Sounio code
>>> result = sounio.run_code('fn main() with IO { print_f64(42.0) }')
>>> result.exit_code
0
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Sounio Team"

# ---------------------------------------------------------------------------
# Core epistemic type — prefer the Rust native extension for speed, fall back
# to the pure Python implementation.
# ---------------------------------------------------------------------------

# Try the Rust/PyO3 native extension first (fast, full GUM).
# Fall back to pure-Python implementation when native is not built.
_NATIVE_BACKEND = False
try:
    from ._sounio_native import (  # type: ignore[import]
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
    from ._knowledge import Knowledge, measure, confidence_gate  # type: ignore[assignment]
    from ._epistemic import EpistemicResult, GUMPropagation      # type: ignore[assignment]
    SensitivityCoefficient = None  # available after `maturin develop`

from ._compile import compile as compile_sio, run as run_sio

# ---------------------------------------------------------------------------
# Provenance chain
# ---------------------------------------------------------------------------

from .provenance import ProvenanceChain, ProvenanceNode  # noqa: F401

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

from ._executor import SounioExecutor, ExecutionResult, CheckResult  # noqa: F401

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

from .types import (  # noqa: F401
    Molecule,
    PKParameters,
    PatientData,
    SimulationResult,
    ScreeningResult,
    PipelineResult,
)

# ---------------------------------------------------------------------------
# Ontology integration
# ---------------------------------------------------------------------------

from . import ontology  # noqa: F401
from .ontology import (  # noqa: F401
    OntologyMapping,
    OntologyResolver,
    ResolvedOntologyTerm,
    ancestors,
    clinical_normalize,
    is_subclass,
    map_term,
    resolve,
    search,
)

# ---------------------------------------------------------------------------
# Module-level convenience functions (use a shared default executor)
# ---------------------------------------------------------------------------

_default_executor: "SounioExecutor | None" = None


def get_executor(**kwargs) -> SounioExecutor:
    """Return the module-level default SounioExecutor (created lazily)."""
    global _default_executor
    if _default_executor is None:
        _default_executor = SounioExecutor(**kwargs)
    return _default_executor


def reset_executor() -> None:
    """Discard the cached default executor (useful in tests)."""
    global _default_executor
    _default_executor = None


def run_file(path: str, timeout: int = 30, **kwargs) -> ExecutionResult:
    """JIT-run a .sio file. See :class:`SounioExecutor.run_file`."""
    return get_executor().run_file(path, timeout=timeout)


def run_code(code: str, timeout: int = 30, **kwargs) -> ExecutionResult:
    """Run inline Sounio code. See :class:`SounioExecutor.run_code`."""
    return get_executor().run_code(code, timeout=timeout)


def check_file(path: str, **kwargs) -> CheckResult:
    """Type-check a .sio file. See :class:`SounioExecutor.check_file`."""
    return get_executor().check_file(path, **kwargs)


async def async_run_file(path: str, timeout: float = 60.0) -> ExecutionResult:
    """Async JIT-run a .sio file. See :meth:`SounioExecutor.async_run_file`."""
    return await get_executor().async_run_file(path, timeout=timeout)


async def async_run_code(code: str, timeout: float = 60.0) -> ExecutionResult:
    """Async run inline Sounio code. See :meth:`SounioExecutor.async_run_code`."""
    return await get_executor().async_run_code(code, timeout=timeout)


# ---------------------------------------------------------------------------
# Kernel client
# ---------------------------------------------------------------------------

from .kernel_client import KernelConnection, KernelResult, launch_jupyter_kernel  # noqa: F401

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

try:
    from .dashboard import serve_dashboard, create_app  # noqa: F401
    _HAS_DASHBOARD = True
except ImportError:
    _HAS_DASHBOARD = False

# ---------------------------------------------------------------------------
# Optional integrations
# ---------------------------------------------------------------------------

try:
    from .integrations.numpy_ext import UncertainArray  # noqa: F401
    _HAS_NUMPY_EXT = True
except ImportError:
    _HAS_NUMPY_EXT = False

try:
    from .integrations.pandas_ext import EpistemicDataFrame  # noqa: F401
    _HAS_PANDAS_EXT = True
except ImportError:
    _HAS_PANDAS_EXT = False

# Integrations module
from . import integrations  # noqa: F401

# High-level modules
try:
    from . import report  # noqa: F401
    from .report import ReportBuilder  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core
    "Knowledge",
    "PureKnowledge",
    # Provenance
    "ProvenanceChain",
    "ProvenanceNode",
    # Executor
    "SounioExecutor",
    "ExecutionResult",
    "CheckResult",
    # Domain types
    "Molecule",
    "PKParameters",
    "PatientData",
    "SimulationResult",
    "ScreeningResult",
    "PipelineResult",
    # Ontology
    "ontology",
    "ResolvedOntologyTerm",
    "OntologyMapping",
    "OntologyResolver",
    "resolve",
    "search",
    "ancestors",
    "is_subclass",
    "map_term",
    "clinical_normalize",
    # Integrations
    "integrations",
    # Report generation
    "report",
    "ReportBuilder",
    # Kernel client
    "KernelConnection",
    "KernelResult",
    "launch_jupyter_kernel",
    # Dashboard
    "serve_dashboard",
    "create_app",
    # Convenience
    "get_executor",
    "reset_executor",
    "run_file",
    "run_code",
    "check_file",
    "async_run_file",
    "async_run_code",
    # Meta
    "__version__",
    "_NATIVE",
]
