"""pytest configuration for sounio-py tests.

Adds ``python/`` to sys.path so tests can import the pure-Python package
without needing a maturin build.
"""

import sys
import os

# Resolve absolute path to the python/ source directory
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "python")
_PKG_ROOT = os.path.normpath(_PKG_ROOT)

if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
