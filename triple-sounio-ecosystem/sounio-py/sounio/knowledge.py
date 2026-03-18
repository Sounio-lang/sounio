"""Redirect: load knowledge.py from python/sounio/knowledge.py."""
import importlib.util as _util
import os as _os

_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "python", "sounio", "knowledge.py")
_spec = _util.spec_from_file_location("_sounio_knowledge_impl", _path)
_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Knowledge = _mod.Knowledge

__all__ = ["Knowledge"]
