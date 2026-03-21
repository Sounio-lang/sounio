"""Sounio integrations with scientific Python ecosystem."""

# Always available
from .numpy_ext import UncertainArray
from .pandas_ext import EpistemicDataFrame

# Optional integrations (graceful import)
try:
    from . import plotting
except ImportError:
    plotting = None

try:
    from . import pubchem
except ImportError:
    pubchem = None

try:
    from . import rdkit_ext
except ImportError:
    rdkit_ext = None

try:
    from . import clinical
except ImportError:
    clinical = None

__all__ = [
    "UncertainArray", "EpistemicDataFrame",
    "plotting", "pubchem", "rdkit_ext", "clinical",
]
