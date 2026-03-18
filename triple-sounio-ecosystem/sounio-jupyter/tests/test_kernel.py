"""Tests for the Sounio kernel."""

import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSounioKernelConstants:
    """Test Sounio kernel constants and class attributes (without instantiation)."""

    def test_kernel_class_exists(self):
        """Test that kernel module can be parsed."""
        import ast

        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "kernel.py")) as f:
            tree = ast.parse(f.read())
            class_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "SounioKernel":
                    class_found = True
                    break
            assert class_found, "SounioKernel class should exist"

    def test_kernel_attributes_in_source(self):
        """Test kernel has required attributes in source."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "kernel.py")) as f:
            source = f.read()
            assert 'language = "sounio"' in source
            assert "implementation" in source
            assert "language_info" in source
            assert "banner" in source

    def test_kernel_methods_in_source(self):
        """Test kernel has required methods in source."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "kernel.py")) as f:
            source = f.read()
            assert "def do_execute" in source
            assert "def do_complete" in source
            assert "def do_inspect" in source
            assert "def do_is_complete" in source
            assert "def do_shutdown" in source


class TestCodeWrapping:
    """Test code wrapping functionality."""

    def test_wrap_expression(self):
        """Test wrapping simple expressions."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "executor",
            os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "executor.py"),
        )
        executor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(executor_module)
        CellExecutor = executor_module.CellExecutor

        executor = CellExecutor()
        wrapped = executor._wrap_code("let x = 1 + 1")
        assert "fn main()" in wrapped
        assert "let x = 1 + 1" in wrapped

    def test_wrap_preserves_functions(self):
        """Test that function definitions are not wrapped."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "executor",
            os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "executor.py"),
        )
        executor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(executor_module)
        CellExecutor = executor_module.CellExecutor

        executor = CellExecutor()
        code = "fn helper() -> i32 {\n    42\n}"
        wrapped = executor._wrap_code(code)
        assert wrapped == code

    def test_wrap_preserves_type_defs(self):
        """Test that type definitions are not wrapped."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "executor",
            os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "executor.py"),
        )
        executor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(executor_module)
        CellExecutor = executor_module.CellExecutor

        executor = CellExecutor()
        code = "type MyInt = i32"
        wrapped = executor._wrap_code(code)
        assert wrapped == code


class TestCompletionImport:
    """Test autocompletion module (imports directly to avoid ipykernel)."""

    def test_completion_module_exists(self):
        """Test completion module can be parsed."""
        import ast

        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "completion.py")) as f:
            tree = ast.parse(f.read())
            class_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "SounioCompleter":
                    class_found = True
                    break
            assert class_found, "SounioCompleter class should exist"

    def test_completion_keywords_defined(self):
        """Test that Sounio keywords are defined."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "completion.py")) as f:
            source = f.read()
            assert "SOUNIO_KEYWORDS" in source
            assert '"let"' in source
            assert '"fn"' in source
            assert '"struct"' in source


class TestMagicsImport:
    """Test magics module (imports directly to avoid ipykernel)."""

    def test_magics_module_exists(self):
        """Test magics module can be parsed."""
        import ast

        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "magics.py")) as f:
            tree = ast.parse(f.read())
            class_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "SounioMagics":
                    class_found = True
                    break
            assert class_found, "SounioMagics class should exist"

    def test_magics_methods_defined(self):
        """Test that magic command handlers are defined."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "magics.py")) as f:
            source = f.read()
            assert "def magic_time" in source
            assert "def magic_timeit" in source
            assert "def magic_sounio" in source
            assert "def handle_magic" in source


if __name__ == "__main__":
    import pytest

    # Run tests if pytest is available
    pytest.main([__file__, "-v"])
