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

try:
    from ._sounio_native import Knowledge as _NativeKnowledge
    Knowledge = _NativeKnowledge
    _NATIVE = True
except ImportError:
    from .knowledge import Knowledge  # type: ignore[assignment]
    _NATIVE = False

# Always expose the pure-Python module so users can import it explicitly.
from .knowledge import Knowledge as PureKnowledge  # noqa: F401

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
    # Convenience
    "get_executor",
    "reset_executor",
    "run_file",
    "run_code",
    "check_file",
    # Meta
    "__version__",
    "_NATIVE",
]
