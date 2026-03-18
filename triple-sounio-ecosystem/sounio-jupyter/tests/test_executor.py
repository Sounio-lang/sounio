"""Tests for CellExecutor."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the executor directly to avoid ipykernel dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "executor",
    os.path.join(os.path.dirname(__file__), '..', 'sounio_kernel', 'executor.py')
)
executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor_module)
CellExecutor = executor_module.CellExecutor


def test_executor_initialization():
    """Test that CellExecutor initializes correctly."""
    executor = CellExecutor()
    assert executor.souc_binary is not None, "souc binary should be found"
    assert executor.stdlib_path is not None, "stdlib path should be found"
    executor.cleanup()


def test_code_wrapping_expressions():
    """Test that expressions are wrapped in main function."""
    executor = CellExecutor()

    # Test simple expression
    wrapped = executor._wrap_code("let x = 1 + 1")
    assert "fn main()" in wrapped
    assert "let x = 1 + 1" in wrapped

    # Test variable declaration
    wrapped = executor._wrap_code("var y = 5")
    assert "fn main()" in wrapped

    executor.cleanup()


def test_code_wrapping_preserves_functions():
    """Test that function definitions are not wrapped."""
    executor = CellExecutor()

    code = """fn helper() -> i32 {
    42
}"""
    wrapped = executor._wrap_code(code)
    assert wrapped == code, "Functions should not be wrapped"

    executor.cleanup()


def test_code_wrapping_preserves_type_defs():
    """Test that type definitions are not wrapped."""
    executor = CellExecutor()

    code = "type MyInt = i32"
    wrapped = executor._wrap_code(code)
    assert wrapped == code, "Type definitions should not be wrapped"

    executor.cleanup()


if __name__ == "__main__":
    test_executor_initialization()
    test_code_wrapping_expressions()
    test_code_wrapping_preserves_functions()
    test_code_wrapping_preserves_type_defs()
    print("All executor tests passed!")
