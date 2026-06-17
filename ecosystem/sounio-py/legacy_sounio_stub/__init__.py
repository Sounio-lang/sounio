"""
sounio: Python bindings for the Sounio programming language.

NOTE: This file is the legacy stub. The authoritative package lives in
``python/sounio/``.  When built with maturin the ``python/`` tree is used
and this file is not part of the installed wheel.

For development without a maturin build, add ``python/`` to your PYTHONPATH
or use pytest (which does this automatically via pyproject.toml pythonpath).
"""

# Attempt to load the native extension; fall back to pure Python.
try:
    from ._sounio_native import Knowledge, knowledge_add, knowledge_mul  # noqa: F401
    _NATIVE = True
except ImportError:
    # Pure-Python fallback: load knowledge.py directly from python/sounio/
    import sys, os as _os, importlib.util as _util
    _kmod_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)), "python", "sounio", "knowledge.py"
    )
    _spec = _util.spec_from_file_location("_sounio_knowledge_fallback", _kmod_path)
    _kmod = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_kmod)
    Knowledge = _kmod.Knowledge  # type: ignore[assignment]  # noqa: F811
    knowledge_add = lambda a, b: a + b  # noqa: E731
    knowledge_mul = lambda a, b: a * b  # noqa: E731
    _NATIVE = False

__version__ = "0.1.0"
__author__ = "Sounio Team"

__all__ = [
    "Knowledge",
    "knowledge_add",
    "knowledge_mul",
    "__version__",
]
